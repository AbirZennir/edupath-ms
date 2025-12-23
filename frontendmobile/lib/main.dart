import 'package:flutter/material.dart';
import 'screens/login_page.dart';
import 'screens/register_page.dart';
import 'screens/main_shell.dart';
import 'welcome_page.dart';
import 'preferences_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = PreferencesService();
  await prefs.loadSettings();
  runApp(const EduPathApp());
}

class EduPathApp extends StatelessWidget {
  const EduPathApp({super.key});

  @override
  Widget build(BuildContext context) {
    final prefs = PreferencesService();

    return AnimatedBuilder(
      animation: prefs,
      builder: (context, child) {
        return MaterialApp(
          title: 'EduPath',
          debugShowCheckedModeBanner: false,
          themeMode: prefs.isDarkMode ? ThemeMode.dark : ThemeMode.light,
          theme: ThemeData(
            useMaterial3: true,
            scaffoldBackgroundColor: const Color(0xFFF6F6F6),
            primaryColor: const Color(0xFF2F65D9),
            colorScheme: ColorScheme.fromSeed(
              seedColor: const Color(0xFF2F65D9),
              primary: const Color(0xFF2F65D9),
              secondary: const Color(0xFF22C55E),
              surface: Colors.white,
              brightness: Brightness.light,
            ),
            appBarTheme: const AppBarTheme(
              backgroundColor: Colors.white,
              elevation: 0,
              foregroundColor: Colors.black87,
            ),
            inputDecorationTheme: InputDecorationTheme(
              filled: true,
              fillColor: Colors.white,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: const BorderSide(color: Color(0xFFE8EAEE)),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: const BorderSide(color: Color(0xFFE8EAEE)),
              ),
            ),
             cardTheme: CardThemeData(
              color: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            ),
          ),
          darkTheme: ThemeData(
            useMaterial3: true,
            scaffoldBackgroundColor: const Color(0xFF111827),
            primaryColor: const Color(0xFF3B82F6),
            colorScheme: ColorScheme.fromSeed(
              seedColor: const Color(0xFF3B82F6),
              primary: const Color(0xFF3B82F6),
              secondary: const Color(0xFF22C55E),
              surface: const Color(0xFF1F2937),
              brightness: Brightness.dark,
            ),
            appBarTheme: const AppBarTheme(
              backgroundColor: Color(0xFF1F2937),
              elevation: 0,
              foregroundColor: Colors.white,
            ),
            inputDecorationTheme: InputDecorationTheme(
              filled: true,
              fillColor: const Color(0xFF374151),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide.none,
              ),
              hintStyle: const TextStyle(color: Color(0xFF9CA3AF)),
            ),
            cardTheme: CardThemeData(
              color: const Color(0xFF1F2937),
              elevation: 0,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            ),
            textTheme: const TextTheme(
              bodyMedium: TextStyle(color: Colors.white),
              bodyLarge: TextStyle(color: Colors.white),
              titleLarge: TextStyle(color: Colors.white),
            ),
            iconTheme: const IconThemeData(color: Colors.white70),
          ),
          initialRoute: '/welcome',
          routes: {
            '/welcome': (_) => const WelcomePage(),
            '/login': (_) => const LoginPage(),
            '/register': (_) => const RegisterPage(),
            '/app': (_) => const MainShell(),
          },
        );
      },
    );
  }
}
