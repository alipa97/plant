import 'package:flutter/material.dart';
import 'pages/plant_scanner.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'TOGA PLANTS',
      theme: ThemeData(
        primarySwatch: Colors.green,
        scaffoldBackgroundColor: const Color(0xFFF8EFE2), // Cream/beige background
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // Top App Bar
            // Padding(
            //   padding: const EdgeInsets.symmetric(horizontal: 16.0),
            //   child: Row(
            //     mainAxisAlignment: MainAxisAlignment.spaceBetween,
            //     children: [
            //       // Settings icon
            //       IconButton(
            //         icon: const Icon(Icons.settings, color: Colors.black),
            //         onPressed: () {},
            //       ),
            //       // Center logo
            //       Image.asset(
            //         'assets/leaf.png', // You'll need to add this image asset
            //         height: 24,
            //         width: 24,
            //       ),
            //       // Notification and profile icons
            //       Row(
            //         children: [
            //           IconButton(
            //             icon: const Icon(Icons.notifications_outlined, color: Colors.black),
            //             onPressed: () {},
            //           ),
            //           IconButton(
            //             icon: const Icon(Icons.person_outline, color: Colors.black),
            //             onPressed: () {},
            //           ),
            //         ],
            //       ),
            //     ],
            //   ),
            // ),
            
            const SizedBox(height: 16),
            
            // Welcome Banner with Gradient
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Container(
                width: double.infinity,
                height: 145,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [
                      Color(0xFF3C8046), // Darker green
                      Color(0xFF67AE73), // Lighter green
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    // Ellipse decorative elements
                    // Positioned(
                    //   top: 10,
                    //   left: 20,
                    //   child: Image.asset(
                    //     "assets/plant.png",
                    //     width: 90,
                    //     height: 11,
                    //   ),
                    // ),
                    Positioned(
                      right: -40,
                      bottom: -40,
                      child: Container(
                        width: 152,
                        height: 152,
                        decoration: BoxDecoration(
                          color: Colors.transparent,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withOpacity(0.1), width: 1),
                        ),
                      ),
                    ),
                    
                    // Text content
                    Padding(
                      padding: const EdgeInsets.only(left: 20, top: 20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: const [
                          Text(
                            "Welcome",
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w400,
                              color: Colors.white,
                            ),
                          ),
                          SizedBox(height: 8),
                          Text(
                            "GET TO KNOW\nTOGA PLANTS",
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                          ),
                          SizedBox(height: 8),
                          Text(
                            "Check it out",
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w300,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                    
                    // Plant image
                    Align(
                      alignment: Alignment.centerRight,
                      child: Image.asset(
                        "assets/plant.png",
                        width: 200, // atur lebarnya sesuai kebutuhan
                        height: double.infinity,
                        fit: BoxFit.cover,
                      ),
                    )

                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Scan Button
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 16.0),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 4,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: const Center(
                  child: Text(
                    'Please, Click the Scan button!',
                    style: TextStyle(
                      color: Color(0xFF4C9E5F),
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ),
            ),

            // Spacer to push nav bar to bottom
            const Spacer(),
            
            // Bottom Navigation Bar
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
              margin: const EdgeInsets.only(bottom: 24.0),
              child: Container(
                height: 60,
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(32),
                ),
                child: Center(
                  child: GestureDetector(
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => const PlantScanner()),
                      );
                    },
                    child: Container(
                      width: 50,
                      height: 50,
                      decoration: BoxDecoration(
                        color: const Color(0xFF4C9E5F),
                        shape: BoxShape.circle,
                        border: Border.all(color: const Color(0xFFF8EFE2), width: 4),
                      ),
                      child: const Icon(Icons.qr_code_2, color: Colors.white),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}