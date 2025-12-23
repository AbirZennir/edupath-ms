import 'package:flutter/material.dart';
import 'dashboard_page.dart';
import 'courses_page.dart';
import 'assignments_page.dart';
import 'grades_page.dart';
import 'profile_page.dart';
import '../api_client.dart';

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _index = 0;
  
  // Dynamically loaded ID
  int _studentId = 1; 
  final ApiClient _apiClient = ApiClient();
  
  // Student data loaded once and shared across pages
  String _studentName = 'Étudiant';
  String _studentEmail = 'email@edupath.fr';
  bool _dataLoaded = false;

  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    _loadStudentData();
  }

  Future<void> _loadStudentData() async {
    try {
      // First, try to get the real user ID
      final storedId = await _apiClient.getSavedUserId();
      if (storedId != null) {
        _studentId = storedId;
      }
      
      final data = await _apiClient.getDashboard(_studentId);
      final student = data['student'] as Map<String, dynamic>?;
      
      if (!mounted) return;
      setState(() {
        _studentName = '${student?['prenom'] ?? ''} ${student?['nom'] ?? ''}'.trim();
        if (_studentName.isEmpty) _studentName = 'Étudiant';
        final prenom = student?['prenom']?.toString().toLowerCase() ?? 'email';
        final nom = student?['nom']?.toString().toLowerCase() ?? 'user';
        _studentEmail = '$prenom.$nom@edupath.fr';
        _dataLoaded = true;
        _initPages(); // Rebuild pages with new data
      });
    } catch (e) {
      if (!mounted) return;
      // Keep default values if loading fails
      setState(() {
        _dataLoaded = true;
        _initPages();
      });
    }
  }

  void _initPages() {
    _pages = [
      DashboardPage(
        studentId: _studentId,
        onTabChange: (index) => setState(() => _index = index),
      ),
      CoursesPage(studentId: _studentId),
      AssignmentsPage(studentId: _studentId),
      GradesPage(studentId: _studentId),
      ProfilePage(
        studentName: _studentName,
        studentEmail: _studentEmail,
      ),
    ];
  }

  @override
  Widget build(BuildContext context) {
    const inactiveColor = Color(0xFF9CA3AF);
    const activeColor = Color(0xFF2F65D9);

    if (!_dataLoaded) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(color: Color(0xFF2F65D9)),
        ),
      );
    }

    return Scaffold(
      body: _pages[_index],
      bottomNavigationBar: BottomNavigationBar(
        backgroundColor: Colors.white,
        currentIndex: _index,
        onTap: (i) => setState(() => _index = i),
        type: BottomNavigationBarType.fixed,
        selectedItemColor: activeColor,
        unselectedItemColor: inactiveColor,
        showUnselectedLabels: true,
        selectedFontSize: 12,
        unselectedFontSize: 12,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined),
            label: 'Accueil',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.menu_book_outlined),
            label: 'Cours',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.assignment_outlined),
            label: 'Devoirs',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart_outlined),
            label: 'Notes',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline),
            label: 'Profil',
          ),
        ],
      ),
    );
  }
}
