import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart'; // Added for compute
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

// Helper class for passing data to the isolate
class IsolateData {
  final CameraImage cameraImage;
  final int inputSize;

  IsolateData(this.cameraImage, this.inputSize);
}

// Top-level function to be run in an isolate for image processing
Future<List<List<List<List<double>>>>> _processImageInIsolate(IsolateData isolateData) async {
  final cameraImage = isolateData.cameraImage;
  final inputSize = isolateData.inputSize;

  final img.Image? image = _convertYUV420ToImageStatic(cameraImage);
  if (image == null) {
    print('Isolate: Failed to convert camera image');
    // Return an empty or specific error structure if needed, or let it throw.
    // For now, let it throw, it will be caught by the caller of compute.
    throw Exception('Isolate: Failed to convert camera image');
  }

  final resizedImage = img.copyResize(image, width: inputSize, height: inputSize);
  
  var imageInput = List.generate(
    1,
    (_) => List.generate(
      inputSize,
      (y) => List.generate(
        inputSize,
        (x) => List.filled(3, 0.0),
      ),
    ),
  );

  for (int y = 0; y < inputSize; y++) {
    for (int x = 0; x < inputSize; x++) {
      final pixel = resizedImage.getPixel(x, y);
      // Normalize pixel values to [0, 1]
      imageInput[0][y][x][0] = pixel.r / 255.0;
      imageInput[0][y][x][1] = pixel.g / 255.0;
      imageInput[0][y][x][2] = pixel.b / 255.0;
    }
  }
  return imageInput;
}

// Static version of _convertYUV420ToImage for use in isolate
// Note: CameraImage planes (Uint8List) are transferable.
img.Image? _convertYUV420ToImageStatic(CameraImage cameraImage) {
  try {
    final width = cameraImage.width;
    final height = cameraImage.height;

    final yBuffer = cameraImage.planes[0].bytes;
    final uBuffer = cameraImage.planes[1].bytes;
    final vBuffer = cameraImage.planes[2].bytes;
    final int yRowStride = cameraImage.planes[0].bytesPerRow;
    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1; // Handle null with a default

    final image = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int yIndex = y * yRowStride + x;
        
        final int uvx = x ~/ 2;
        final int uvy = y ~/ 2;
        // Calculate the U and V indices using their respective row strides and pixel strides
        final int uIndex = uvy * uvRowStride + uvx * uvPixelStride;
        final int vIndex = uvy * uvRowStride + uvx * uvPixelStride;

        if (yIndex < yBuffer.length && uIndex < uBuffer.length && vIndex < vBuffer.length) {
          final yValue = yBuffer[yIndex];
          final uValue = uBuffer[uIndex];
          final vValue = vBuffer[vIndex];

          // ITU-R BT.601 YUV to RGB conversion
          // Ensure values are doubles for calculation then clamp and round.
          final double yVal = yValue.toDouble();
          final double uVal = uValue.toDouble() - 128.0;
          final double vVal = vValue.toDouble() - 128.0;

          final int r = (yVal + 1.402 * vVal).round().clamp(0, 255);
          final int g = (yVal - 0.344136 * uVal - 0.714136 * vVal).round().clamp(0, 255);
          final int b = (yVal + 1.772 * uVal).round().clamp(0, 255);
          
          image.setPixelRgba(x, y, r, g, b, 255);
        } else {
           // If out of bounds, set to black or handle as an error
          image.setPixelRgba(x, y, 0, 0, 0, 255);
        }
      }
    }
    return image;
  } catch (e) {
    print('Isolate: Error converting YUV420 to Image: $e');
    return null;
  }
}


class PlantScanner extends StatefulWidget {
  const PlantScanner({super.key});

  @override
  State<PlantScanner> createState() => _PlantScannerState();
}

class _PlantScannerState extends State<PlantScanner> {
  CameraController? _controller;
  bool _isCameraInitialized = false;

  Interpreter? _interpreter;
  List<String> _labels = [];
  
  // Pre-allocated input and output tensors
  List<Object>? _inputTensors;
  Map<int, Object>? _outputTensors;

  bool _isDetecting = false;
  bool _isModelLoaded = false;
  List<DetectionResult> _detections = [];
  
  // Untuk mengontrol FPS detection
  DateTime _lastDetection = DateTime.now();
  static const int _detectionIntervalMs = 750; // Deteksi setiap 750ms (previously 500ms)
  
  // Model configuration berdasarkan attributes
  static const int _inputSize = 320; // Ukuran input untuk object detection
  static const int _maxDetections = 10;
  static const double _confidenceThreshold = 0.5;
  static const double _iouThreshold = 0.6;
  static const int _numClasses = 10;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadModelAndLabels();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        print("Tidak ada kamera yang tersedia");
        return;
      }
      
      _controller = CameraController(
        cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      
      await _controller!.initialize();
      
      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
        
        // Mulai realtime detection setelah kamera ready
        _startImageStream();
      }
    } catch (e) {
      print("Error inisialisasi kamera: $e");
    }
  }

  void _startImageStream() {
    if (_controller != null && _isCameraInitialized) {
      _controller!.startImageStream(_processCameraImage);
    }
  }

  Future<void> _loadModelAndLabels() async {
    try {
      final modelOptions = InterpreterOptions();
      // Try to disable GPU delegate and NNAPI to see if it resolves native crashes/nulls
      // modelOptions.addDelegate(GpuDelegateV2()); // Example if we wanted to add GPU
      // Not adding any delegates forces CPU execution, which is more stable for debugging.
      
      _interpreter = await Interpreter.fromAsset(
        'assets/model.tflite',
        options: modelOptions,
      );
      
      print('Model berhasil dimuat');
      print('Input tensors: ${_interpreter!.getInputTensors().length}');
      print('Output tensors: ${_interpreter!.getOutputTensors().length}');
      
      // Print input shapes
      for (int i = 0; i < _interpreter!.getInputTensors().length; i++) {
        print('Input $i shape: ${_interpreter!.getInputTensor(i).shape}');
      }
      
      // Print output shapes
      for (int i = 0; i < _interpreter!.getOutputTensors().length; i++) {
        print('Output $i shape: ${_interpreter!.getOutputTensor(i).shape}');
      }
      
      // Load labels
      try {
        final labelsData = await rootBundle.loadString('assets/labels.txt');
        _labels = labelsData.split('\n')
            .where((label) => label.trim().isNotEmpty)
            .toList();
        
        print('Labels berhasil dimuat: ${_labels.length} labels');
        
        setState(() {
          _isModelLoaded = true;
        });
        
        // Pre-allocate tensors after model is loaded
        _initializeTensors();

      } catch (e) {
        print('Error memuat labels: $e');
        _labels = List.generate(_numClasses, (index) => 'Plant $index');
        setState(() {
          _isModelLoaded = true;
        });
         // Pre-allocate tensors even if labels fail, if model loaded
        if (_interpreter != null) {
          _initializeTensors();
        }
      }
    } catch (e) {
      print('Gagal memuat model: $e');
    }
  }

  void _initializeTensors() {
    if (_interpreter == null) return;

    // Initialize Input Tensor (imageInput)
    // Shape: [1, 320, 320, 3]
    _inputTensors = [
      List.generate(
        1,
        (_) => List.generate(
          _inputSize,
          (y) => List.generate(
            _inputSize,
            (x) => List.filled(3, 0.0), // Placeholder, will be filled
          ),
        ),
      ),
    ];
    print('Input tensors pre-allocated.');

    // Initialize Output Tensors
    // Output 0 shape: [1, 10] (Scores)
    // Output 1 shape: [1, 10, 4] (Boxes)
    // Output 2 shape: [1] (Number of detections)
    // Output 3 shape: [1, 10] (Classes)
    final outputScores = List.generate(1, (_) => List.filled(_maxDetections, 0.0));
    final outputBoxes = List.generate(1, (_) => List.generate(_maxDetections, (_) => List.filled(4, 0.0)));
    final outputNumDetections = List.filled(1, 0.0);
    final outputClasses = List.generate(1, (_) => List.filled(_maxDetections, 0.0));

    _outputTensors = {
      0: outputScores,
      1: outputBoxes,
      2: outputNumDetections,
      3: outputClasses,
    };
    print('Output tensors pre-allocated.');
  }

  void _processCameraImage(CameraImage cameraImage) async {
    // Kontrol FPS detection
    final now = DateTime.now();
    if (now.difference(_lastDetection).inMilliseconds < _detectionIntervalMs) {
      return;
    }
    
    if (_isDetecting || !_isModelLoaded || _interpreter == null) {
      return;
    }

    _isDetecting = true;
    _lastDetection = now;

    try {
      // Konversi CameraImage ke format yang bisa diproses
      final img.Image? image = _convertYUV420ToImage(cameraImage);
      if (image == null) {
        print('Failed to convert camera image');
        return;
      }

      // Resize image untuk model object detection
      final resizedImage = img.copyResize(image, width: _inputSize, height: _inputSize);

      // Fill the pre-allocated input tensor
      _fillInputTensor(resizedImage);

      // Check if tensors are ready (they should be if _isModelLoaded is true)
      if (_inputTensors == null || _outputTensors == null) {
        print("Error: Tensors not initialized before inference.");
        _isDetecting = false;
        return;
      }

      // print('Running inference with pre-allocated tensors...');
      // No need to print number of inputs/outputs as they are fixed now.

      // Run inference using pre-allocated tensors
      try {
        _interpreter!.runForMultipleInputs(_inputTensors!, _outputTensors!); // Use runForMultipleInputs if _inputTensors is List<Object>
        // If _inputTensors![0] is the actual tensor and not a list containing it:
        // _interpreter!.run(_inputTensors![0], _outputTensors!);
        print('Inference completed successfully');
      } catch (e) {
        print('Inference error: $e');
        _isDetecting = false; // Reset detection flag
        return; // Stop processing if inference fails
      }

      // Process hasil deteksi using the pre-allocated _outputTensors
      final detections = _processObjectDetectionOutput(_outputTensors!, cameraImage.width, cameraImage.height);

      if (mounted) {
        setState(() {
          _detections = detections;
        });
      }
    } catch (e) {
      print("Error processing camera image: $e");
      print("Stack trace: ${StackTrace.current}");
    } finally {
      _isDetecting = false;
    }
  }

  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    try {
      final width = cameraImage.width;
      final height = cameraImage.height;
      
      final yBuffer = cameraImage.planes[0].bytes;
      final uBuffer = cameraImage.planes[1].bytes;
      final vBuffer = cameraImage.planes[2].bytes;

      final image = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final yIndex = y * width + x;
          final uvIndex = (y ~/ 2) * (width ~/ 2) + (x ~/ 2);

          if (yIndex < yBuffer.length && uvIndex < uBuffer.length && uvIndex < vBuffer.length) {
            final yValue = yBuffer[yIndex];
            final uValue = uBuffer[uvIndex];
            final vValue = vBuffer[uvIndex];

            final r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
            final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).round().clamp(0, 255);
            final b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);

            image.setPixelRgba(x, y, r, g, b, 255);
          }
        }
      }
      return image;
    } catch (e) {
      print('Error converting YUV420 to Image: $e');
      return null;
    }
  }

// Fills the pre-allocated _inputTensors
void _fillInputTensor(img.Image image) {
  if (_inputTensors == null || _inputTensors![0] == null) {
    print("Error: Input tensors not pre-allocated for filling.");
    return;
  }

  try {
    var imageInput = _inputTensors![0] as List<List<List<List<double>>>>;

    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = image.getPixel(x, y);
        imageInput[0][y][x][0] = pixel.r.toDouble() / 255.0;
        imageInput[0][y][x][1] = pixel.g.toDouble() / 255.0;
        imageInput[0][y][x][2] = pixel.b.toDouble() / 255.0;
      }
    }
  } catch (e) {
    print('Error filling input tensor: $e');
  }
}

  List<List<double>> _generateAnchors() {
    // Generate anchor boxes untuk object detection
    // Ini adalah implementasi sederhana, mungkin perlu disesuaikan dengan model spesifik
    List<List<double>> anchors = [];
    
    // Generate anchors untuk berbagai scale dan aspect ratio
    final scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9];
    final aspectRatios = [1.0, 2.0, 0.5];
    
    for (int i = 0; i < 6; i++) { // 6 feature maps
      final featureMapSize = _inputSize ~/ math.pow(2, i + 1);
      for (int y = 0; y < featureMapSize; y++) {
        for (int x = 0; x < featureMapSize; x++) {
          for (final scale in scales) {
            for (final aspectRatio in aspectRatios) {
              final centerX = (x + 0.5) / featureMapSize;
              final centerY = (y + 0.5) / featureMapSize;
              final width = scale * math.sqrt(aspectRatio);
              final height = scale / math.sqrt(aspectRatio);
              
              anchors.add([centerY, centerX, height, width]);
              
              if (anchors.length >= 1917) break; // Batasi jumlah anchors
            }
            if (anchors.length >= 1917) break;
          }
          if (anchors.length >= 1917) break;
        }
        if (anchors.length >= 1917) break;
      }
      if (anchors.length >= 1917) break;
    }
    
    // Pad atau trim ke ukuran yang tepat jika diperlukan
    while (anchors.length < 1917) {
      anchors.add([0.5, 0.5, 0.1, 0.1]); // Default anchor
    }
    if (anchors.length > 1917) {
      anchors = anchors.sublist(0, 1917);
    }
    
    return anchors;
  }

  // _prepareOutputs() is removed as outputs are now pre-allocated in _initializeTensors 
  // and stored in _outputTensors. The _outputTensors map is directly passed to interpreter.run().

  List<DetectionResult> _processObjectDetectionOutput(
    Map<int, Object> outputs, 
    int imageWidth, 
    int imageHeight
  ) {
    List<DetectionResult> results = [];
    
    try {
      print('Processing outputs...');
      print('Available output keys: ${outputs.keys.toList()}');
      
      // Safely extract outputs
      if (!outputs.containsKey(0) || !outputs.containsKey(1) || 
          !outputs.containsKey(2) || !outputs.containsKey(3)) {
        print('Missing required output keys');
        return results;
      }
      
      // Based on the new output structure from _prepareOutputs
      final scores = outputs[0] as List;
      final boxes = outputs[1] as List;
      final numDetections = outputs[2] as List;
      final classes = outputs[3] as List;
      
      print('Output shapes:');
      print('Boxes: ${boxes.length} x ${(boxes.isNotEmpty ? (boxes[0] as List).length : 0)} x ${(boxes.isNotEmpty && (boxes[0] as List).isNotEmpty ? (boxes[0][0]as List).length : 0)}');
      print('Classes: ${classes.length} x ${(classes.isNotEmpty ? (classes[0] as List).length : 0)}');
      print('Scores: ${scores.length} x ${(scores.isNotEmpty ? (scores[0] as List).length : 0)}');
      print('NumDetections: ${numDetections.length}');
      
      if (boxes.isEmpty || classes.isEmpty || scores.isEmpty || numDetections.isEmpty) {
        print('Empty output arrays');
        return results;
      }
      
      // numDetections is List<double> with one element, e.g., [10.0]
      final numDet = (numDetections[0] as double).toInt().clamp(0, _maxDetections);
      print('Number of detections: $numDet');
      
      if (numDet == 0) {
        return results;
      }
      
      // scores: List<List<double>> e.g. [[0.9, 0.8, ...]]
      // classes: List<List<double>> e.g. [[1.0, 0.0, ...]]
      // boxes: List<List<List<double>>> e.g. [[[0.1,0.2,0.3,0.4], ...]]
      final scoresBatch = scores[0] as List<double>;
      final classesBatch = classes[0] as List<double>;
      final boxesBatch = boxes[0] as List<List<double>>; // This is List<List<double>>
      
      for (int i = 0; i < numDet; i++) { // Iterate up to numDet
        try {
          // Add bounds checks for safety, though numDet should respect this.
          // However, the model might return inconsistent data.
          if (i >= scoresBatch.length || i >= classesBatch.length || i >= boxesBatch.length) {
            print('Warning: Detection index $i is out of bounds for scores/classes/boxes arrays. Skipping.');
            continue;
          }

          final score = scoresBatch[i];
          final classId = classesBatch[i].toInt();
          
          print('Detection $i: score=$score, classId=$classId');
          
          if (score > _confidenceThreshold && classId >= 0 && classId < _labels.length) {
            final box = boxesBatch[i]; // This is List<double>
            
            if (box.length >= 4) {
              // Model output: ymin, xmin, ymax, xmax
              double y1 = box[0].clamp(0.0, 1.0);
              double x1 = box[1].clamp(0.0, 1.0);
              double y2 = box[2].clamp(0.0, 1.0);
              double x2 = box[3].clamp(0.0, 1.0);
              
              // Convert to pixel coordinates
              final left = (x1 * imageWidth).clamp(0.0, imageWidth.toDouble());
              final top = (y1 * imageHeight).clamp(0.0, imageHeight.toDouble());
              final right = (x2 * imageWidth).clamp(0.0, imageWidth.toDouble());
              final bottom = (y2 * imageHeight).clamp(0.0, imageHeight.toDouble());
              
              // Validate bounding box
              if (right > left && bottom > top && 
                  (right - left) > 10 && (bottom - top) > 10) {
                
                String formattedLabel = classId < _labels.length ? 
                    _labels[classId]
                        .replaceAll('_', ' ')
                        .split(' ')
                        .map((word) => word.isNotEmpty
                            ? '${word[0].toUpperCase()}${word.substring(1).toLowerCase()}'
                            : '')
                        .join(' ') : 'Unknown';
                    
                results.add(DetectionResult(
                  label: formattedLabel,
                  confidence: score,
                  boundingBox: Rect.fromLTRB(left, top, right, bottom),
                ));
                
                print('Added detection: $formattedLabel (${(score * 100).toStringAsFixed(1)}%)');
              }
            } else {
              print('Warning: Box data for detection $i has length ${box.length}, expected >= 4.');
            }
          }
        } catch (e) {
          print('Error processing detection $i: $e');
          continue;
        }
      }
      
      print('Total valid detections: ${results.length}');
      
    } catch (e) {
      print('Error processing object detection output: $e');
      print('Stack trace: ${StackTrace.current}');
    }
    
    // Sort berdasarkan confidence (tertinggi dulu)
    results.sort((a, b) => b.confidence.compareTo(a.confidence));
    
    return results;
  }

  @override
  void dispose() {
    _controller?.stopImageStream();
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  Widget _buildDetectionBox(DetectionResult detection, Size screenSize) {
    if (_controller == null) return Container();
    
    // Calculate scale factors
    final double scaleX = screenSize.width / _controller!.value.previewSize!.height;
    final double scaleY = screenSize.height / _controller!.value.previewSize!.width;
    
    // Scale bounding box
    final scaledRect = Rect.fromLTRB(
      detection.boundingBox.left * scaleX,
      detection.boundingBox.top * scaleY,
      detection.boundingBox.right * scaleX,
      detection.boundingBox.bottom * scaleY,
    );
    
    return Positioned(
      left: scaledRect.left,
      top: scaledRect.top,
      child: Container(
        width: scaledRect.width,
        height: scaledRect.height,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.green, width: 3),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Stack(
          children: [
            // Label background
            Positioned(
              top: -30,
              left: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.green,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${detection.label} ${(detection.confidence * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized || _controller == null) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(color: Colors.green),
              const SizedBox(height: 20),
              Text(
                'Memulai kamera...',
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // Camera Preview
              CameraPreview(_controller!),
              
              // Status indicator
              Positioned(
                top: 50,
                right: 20,
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: _isModelLoaded ? Colors.green : Colors.orange,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 4,
                      ),
                    ],
                  ),
                  child: Icon(
                    _isModelLoaded ? Icons.check : Icons.hourglass_empty,
                    color: Colors.white,
                    size: 16,
                  ),
                ),
              ),

              // Detection boxes
              ...(_detections.map((detection) {
                return _buildDetectionBox(detection, constraints.biggest);
              }).toList()),

              // Detection count / name
              if (_detections.isNotEmpty)
                Positioned(
                  top: 100,
                  left: 20,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.7),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      // Display the name of the highest confidence detection
                      _detections.isNotEmpty
                          ? '${_detections[0].label} terdeteksi'
                          : '${_detections.length} tanaman terdeteksi', // Fallback, though covered by outer if
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),

              // Scanning indicator
              if (_detections.isEmpty && _isModelLoaded)
                Positioned(
                  bottom: 100,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.6),
                        borderRadius: BorderRadius.circular(25),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Text(
                            'Mencari tanaman...',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

class DetectionResult {
  final String label;
  final double confidence;
  final Rect boundingBox;

  DetectionResult({
    required this.label,
    required this.confidence,
    required this.boundingBox,
  });
}