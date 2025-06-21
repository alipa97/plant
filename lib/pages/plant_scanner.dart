import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

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
  
  bool _isDetecting = false;
  bool _isModelLoaded = false;
  List<DetectionResult> _detections = [];
  
  // Untuk mengontrol FPS detection
  DateTime _lastDetection = DateTime.now();
  static const int _detectionIntervalMs = 500; // Deteksi setiap 500ms
  
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
      } catch (e) {
        print('Error memuat labels: $e');
        _labels = List.generate(_numClasses, (index) => 'Plant $index');
        setState(() {
          _isModelLoaded = true;
        });
      }
    } catch (e) {
      print('Gagal memuat model: $e');
    }
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

      // Prepare inputs dengan error handling
      final inputs = _prepareInputs(resizedImage);
      final outputs = _prepareOutputs();

      print('Running inference...');
      print('Number of inputs: ${inputs.length}');
      print('Number of outputs: ${outputs.length}');

      // Run inference dengan multiple inputs
      try {
        _interpreter!.runForMultipleInputs(inputs, outputs);
        print('Inference completed successfully');
      } catch (e) {
        print('Inference error: $e');
        // Coba dengan single input jika multiple inputs gagal
        try {
          print('Trying single input...');
          _interpreter!.run(inputs[0], outputs);
          print('Single input inference successful');
        } catch (e2) {
          print('Single input also failed: $e2');
          return;
        }
      }

      // Process hasil deteksi
      final detections = _processObjectDetectionOutput(outputs, cameraImage.width, cameraImage.height);

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

List<Object> _prepareInputs(img.Image image) {
  try {
    // Input 1: concat - Image tensor [1, 224, 224, 3]
    final imageInput = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = image.getPixel(x, y);
            return [
              pixel.r.toDouble() / 255.0,
              pixel.g.toDouble() / 255.0,
              pixel.b.toDouble() / 255.0,
            ];
          },
        ),
      ),
    );

    // Input 2: convert_scores - [1, 12544, 10]
    final scoresInput = List.generate(
      1,
      (_) => List.generate(
        12544,
        (_) => List.filled(10, 0.0),
      ),
    );

    // Input 3: anchors - [1, 12544, 4]
    final anchorsInput = List.generate(
      1,
      (_) => List.generate(
        12544,
        (_) => [0.5, 0.5, 0.1, 0.1],
      ),
    );

    print('Input shapes prepared:');
    print('Image input: ${imageInput.length}x${imageInput[0].length}x${imageInput[0][0].length}x${imageInput[0][0][0].length}');
    print('Scores input: ${scoresInput.length}x${scoresInput[0].length}x${scoresInput[0][0].length}');
    print('Anchors input: ${anchorsInput.length}x${anchorsInput[0].length}x${anchorsInput[0][0].length}');

    return [imageInput, scoresInput, anchorsInput];
  } catch (e) {
    print('Error preparing inputs: $e');
    rethrow;
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

  Map<int, Object> _prepareOutputs() {
    try {
      // Berdasarkan model dengan 4 output:
      // Buat output buffers yang aman
      final output0 = List.generate(1, (_) => List.generate(_maxDetections, (_) => List.filled(4, 0.0)));
      final output1 = List.generate(1, (_) => List.filled(_maxDetections, 0.0));
      final output2 = List.generate(1, (_) => List.filled(_maxDetections, 0.0));
      final output3 = List.filled(1, 0.0);
      
      print('Output shapes prepared:');
      print('Output 0: ${output0.length}x${output0[0].length}x${(output0[0][0] as List).length}');
      print('Output 1: ${output1.length}x${output1[0].length}');
      print('Output 2: ${output2.length}x${output2[0].length}');
      print('Output 3: ${output3.length}');
      
      return {
        0: output0, // Boxes
        1: output1, // Classes
        2: output2, // Scores
        3: output3, // Number of detections
      };
    } catch (e) {
      print('Error preparing outputs: $e');
      rethrow;
    }
  }

  List<DetectionResult> _processObjectDetectionOutput(
    Map<int, Object> outputs, 
    int imageWidth, 
    int imageHeight
  ) {
    List<DetectionResult> results = [];
    
    try {
      print('Processing outputs...');
      print('Available output keys: ${outputs.keys.toList()}');
      
      // Safely extract outputs dengan null checking
      if (!outputs.containsKey(0) || !outputs.containsKey(1) || 
          !outputs.containsKey(2) || !outputs.containsKey(3)) {
        print('Missing required output keys');
        return results;
      }
      
      final boxes = outputs[0] as List;
      final classes = outputs[1] as List; 
      final scores = outputs[2] as List;
      final numDetections = outputs[3] as List;
      
      print('Output shapes:');
      print('Boxes: ${boxes.length} x ${(boxes.isNotEmpty ? (boxes[0] as List).length : 0)}');
      print('Classes: ${classes.length} x ${(classes.isNotEmpty ? (classes[0] as List).length : 0)}');
      print('Scores: ${scores.length} x ${(scores.isNotEmpty ? (scores[0] as List).length : 0)}');
      print('NumDetections: ${numDetections.length}');
      
      if (boxes.isEmpty || classes.isEmpty || scores.isEmpty || numDetections.isEmpty) {
        print('Empty output arrays');
        return results;
      }
      
      final numDet = (numDetections[0] as double).toInt().clamp(0, _maxDetections);
      print('Number of detections: $numDet');
      
      if (numDet == 0) {
        return results;
      }
      
      final boxesBatch = boxes[0] as List;
      final classesBatch = classes[0] as List;
      final scoresBatch = scores[0] as List;
      
      for (int i = 0; i < numDet && i < boxesBatch.length && i < classesBatch.length && i < scoresBatch.length; i++) {
        try {
          final score = scoresBatch[i] as double;
          final classId = (classesBatch[i] as double).toInt();
          
          print('Detection $i: score=$score, classId=$classId');
          
          if (score > _confidenceThreshold && classId >= 0 && classId < _labels.length) {
            final box = boxesBatch[i] as List;
            
            if (box.length >= 4) {
              double y1 = (box[0] as double).clamp(0.0, 1.0);
              double x1 = (box[1] as double).clamp(0.0, 1.0);
              double y2 = (box[2] as double).clamp(0.0, 1.0);
              double x2 = (box[3] as double).clamp(0.0, 1.0);
              
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

              // Detection count
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
                      '${_detections.length} tanaman terdeteksi',
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