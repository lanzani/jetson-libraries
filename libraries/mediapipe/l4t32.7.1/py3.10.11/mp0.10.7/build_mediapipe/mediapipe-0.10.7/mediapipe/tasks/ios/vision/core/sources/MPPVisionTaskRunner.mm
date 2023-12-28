// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionTaskRunner.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskInfo.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPVisionPacketCreator.h"

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/rect.pb.h"

#include <optional>

namespace {
using ::mediapipe::NormalizedRect;
using ::mediapipe::Packet;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::core::PacketsCallback;
}  // namespace

/** Rotation degrees for a 90 degree rotation to the right. */
static const NSInteger kMPPOrientationDegreesRight = -270;

/** Rotation degrees for a 180 degree rotation. */
static const NSInteger kMPPOrientationDegreesDown = -180;

/** Rotation degrees for a 90 degree rotation to the left. */
static const NSInteger kMPPOrientationDegreesLeft = -90;

static NSString *const kTaskPrefix = @"com.mediapipe.tasks.vision";

#define InputPacketMap(imagePacket, normalizedRectPacket)                              \
  {                                                                                    \
    {_imageInStreamName, imagePacket}, { _normRectInStreamName, normalizedRectPacket } \
  }

@interface MPPVisionTaskRunner () {
  MPPRunningMode _runningMode;
  BOOL _roiAllowed;
  std::string _imageInStreamName;
  std::string _normRectInStreamName;
}
@end

@implementation MPPVisionTaskRunner

- (nullable instancetype)initWithTaskInfo:(MPPTaskInfo *)taskInfo
                              runningMode:(MPPRunningMode)runningMode
                               roiAllowed:(BOOL)roiAllowed
                          packetsCallback:(PacketsCallback)packetsCallback
                     imageInputStreamName:(NSString *)imageInputStreamName
                  normRectInputStreamName:(NSString *)normRectInputStreamName
                                    error:(NSError **)error {
  _roiAllowed = roiAllowed;
  _imageInStreamName = imageInputStreamName.cppString;
  _normRectInStreamName = normRectInputStreamName.cppString;

  switch (runningMode) {
    case MPPRunningModeImage:
    case MPPRunningModeVideo: {
      if (packetsCallback) {
        [MPPCommonUtils createCustomError:error
                                 withCode:MPPTasksErrorCodeInvalidArgumentError
                              description:@"The vision task is in image or video mode. The "
                                          @"delegate must not be set in the task's options."];
        return nil;
      }
      break;
    }
    case MPPRunningModeLiveStream: {
      if (!packetsCallback) {
        [MPPCommonUtils
            createCustomError:error
                     withCode:MPPTasksErrorCodeInvalidArgumentError
                  description:
                      @"The vision task is in live stream mode. An object must be set as the "
                      @"delegate of the task in its options to ensure asynchronous delivery of "
                      @"results."];
        return nil;
      }
      break;
    }
    default: {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Unrecognized running mode"];
      return nil;
    }
  }

  _runningMode = runningMode;
  self = [super initWithCalculatorGraphConfig:[taskInfo generateGraphConfig]
                              packetsCallback:packetsCallback
                                        error:error];
  return self;
}

- (std::optional<NormalizedRect>)normalizedRectWithRegionOfInterest:(CGRect)roi
                                                          imageSize:(CGSize)imageSize
                                                   imageOrientation:
                                                       (UIImageOrientation)imageOrientation
                                                              error:(NSError **)error {
  if (!CGRectEqualToRect(roi, CGRectZero) && !_roiAllowed) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"This task doesn't support region-of-interest."];
    return std::nullopt;
  }

  CGRect calculatedRoi = CGRectEqualToRect(roi, CGRectZero) ? CGRectMake(0.0, 0.0, 1.0, 1.0) : roi;

  NormalizedRect normalizedRect;
  normalizedRect.set_x_center(CGRectGetMidX(calculatedRoi));
  normalizedRect.set_y_center(CGRectGetMidY(calculatedRoi));

  int rotationDegrees = 0;
  switch (imageOrientation) {
    case UIImageOrientationUp:
      break;
    case UIImageOrientationRight: {
      rotationDegrees = kMPPOrientationDegreesRight;
      break;
    }
    case UIImageOrientationDown: {
      rotationDegrees = kMPPOrientationDegreesDown;
      break;
    }
    case UIImageOrientationLeft: {
      rotationDegrees = kMPPOrientationDegreesLeft;
      break;
    }
    default:
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:
                    @"Unsupported UIImageOrientation. `imageOrientation` cannot be equal to "
                    @"any of the mirrored orientations "
                    @"(`UIImageOrientationUpMirrored`,`UIImageOrientationDownMirrored`,`"
                    @"UIImageOrientationLeftMirrored`,`UIImageOrientationRightMirrored`)"];
  }

  normalizedRect.set_rotation(rotationDegrees * M_PI / kMPPOrientationDegreesDown);

  // For 90° and 270° rotations, we need to swap width and height.
  // This is due to the internal behavior of ImageToTensorCalculator, which:
  // - first denormalizes the provided rect by multiplying the rect width or height by the image
  //   width or height, respectively.
  // - then rotates this by denormalized rect by the provided rotation, and uses this for cropping,
  // - then finally rotates this back.
  if (rotationDegrees % 180 == 0) {
    normalizedRect.set_width(CGRectGetWidth(calculatedRoi));
    normalizedRect.set_height(CGRectGetHeight(calculatedRoi));
  } else {
    const float width = CGRectGetHeight(calculatedRoi) * imageSize.height / imageSize.width;
    const float height = CGRectGetWidth(calculatedRoi) * imageSize.width / imageSize.height;

    normalizedRect.set_width(width);
    normalizedRect.set_height(height);
  }

  return normalizedRect;
}

- (std::optional<PacketMap>)inputPacketMapWithMPPImage:(MPPImage *)image
                                      regionOfInterest:(CGRect)roi
                                                 error:(NSError **)error {
  std::optional<NormalizedRect> rect =
      [self normalizedRectWithRegionOfInterest:roi
                                     imageSize:CGSizeMake(image.width, image.height)
                              imageOrientation:image.orientation
                                         error:error];
  if (!rect.has_value()) {
    return std::nullopt;
  }

  Packet imagePacket = [MPPVisionPacketCreator createPacketWithMPPImage:image error:error];
  if (imagePacket.IsEmpty()) {
    return std::nullopt;
  }

  Packet normalizedRectPacket =
      [MPPVisionPacketCreator createPacketWithNormalizedRect:rect.value()];

  PacketMap inputPacketMap = InputPacketMap(imagePacket, normalizedRectPacket);
  return inputPacketMap;
}

- (std::optional<PacketMap>)inputPacketMapWithMPPImage:(MPPImage *)image
                                      regionOfInterest:(CGRect)roi
                               timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                                 error:(NSError **)error {
  std::optional<NormalizedRect> rect =
      [self normalizedRectWithRegionOfInterest:roi
                                     imageSize:CGSizeMake(image.width, image.height)
                              imageOrientation:image.orientation
                                         error:error];
  if (!rect.has_value()) {
    return std::nullopt;
  }

  Packet imagePacket = [MPPVisionPacketCreator createPacketWithMPPImage:image
                                                timestampInMilliseconds:timestampInMilliseconds
                                                                  error:error];
  if (imagePacket.IsEmpty()) {
    return std::nullopt;
  }

  Packet normalizedRectPacket =
      [MPPVisionPacketCreator createPacketWithNormalizedRect:rect.value()
                                     timestampInMilliseconds:timestampInMilliseconds];

  PacketMap inputPacketMap = InputPacketMap(imagePacket, normalizedRectPacket);
  return inputPacketMap;
}

- (std::optional<PacketMap>)processImage:(MPPImage *)image
                        regionOfInterest:(CGRect)regionOfInterest
                                   error:(NSError **)error {
  if (_runningMode != MPPRunningModeImage) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The vision task is not initialized with "
                                                     @"image mode. Current Running Mode: %@",
                                                     MPPRunningModeDisplayName(_runningMode)]];
    return std::nullopt;
  }

  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:image
                                                            regionOfInterest:regionOfInterest
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return std::nullopt;
  }

  return [self processPacketMap:inputPacketMap.value() error:error];
}

- (std::optional<PacketMap>)processImage:(MPPImage *)image error:(NSError **)error {
  return [self processImage:image regionOfInterest:CGRectZero error:error];
}

- (std::optional<PacketMap>)processVideoFrame:(MPPImage *)videoFrame
                             regionOfInterest:(CGRect)regionOfInterest
                      timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                        error:(NSError **)error {
  if (_runningMode != MPPRunningModeVideo) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The vision task is not initialized with "
                                                     @"video mode. Current Running Mode: %@",
                                                     MPPRunningModeDisplayName(_runningMode)]];
    return std::nullopt;
  }

  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:videoFrame
                                                            regionOfInterest:regionOfInterest
                                                     timestampInMilliseconds:timestampInMilliseconds
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return std::nullopt;
  }

  return [self processPacketMap:inputPacketMap.value() error:error];
}

- (std::optional<PacketMap>)processVideoFrame:(MPPImage *)videoFrame
                      timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                        error:(NSError **)error {
  return [self processVideoFrame:videoFrame
                regionOfInterest:CGRectZero
         timestampInMilliseconds:timestampInMilliseconds
                           error:error];
}

- (BOOL)processLiveStreamImage:(MPPImage *)image
              regionOfInterest:(CGRect)regionOfInterest
       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                         error:(NSError **)error {
  if (_runningMode != MPPRunningModeLiveStream) {
    [MPPCommonUtils
        createCustomError:error
                 withCode:MPPTasksErrorCodeInvalidArgumentError
              description:[NSString stringWithFormat:@"The vision task is not initialized with "
                                                     @"live stream mode. Current Running Mode: %@",
                                                     MPPRunningModeDisplayName(_runningMode)]];
    return NO;
  }

  std::optional<PacketMap> inputPacketMap = [self inputPacketMapWithMPPImage:image
                                                            regionOfInterest:regionOfInterest
                                                     timestampInMilliseconds:timestampInMilliseconds
                                                                       error:error];
  if (!inputPacketMap.has_value()) {
    return NO;
  }

  return [self sendPacketMap:inputPacketMap.value() error:error];
}

- (BOOL)processLiveStreamImage:(MPPImage *)image
       timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                         error:(NSError **)error {
  return [self processLiveStreamImage:image
                     regionOfInterest:CGRectZero
              timestampInMilliseconds:timestampInMilliseconds
                                error:error];
}

+ (const char *)uniqueDispatchQueueNameWithSuffix:(NSString *)suffix {
  return [NSString stringWithFormat:@"%@.%@_%@", kTaskPrefix, suffix, [NSString uuidString]]
      .UTF8String;
}

@end
