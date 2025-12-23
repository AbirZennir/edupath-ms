import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'course_detail_page.dart';
import '../api_client.dart';

class CoursesPage extends StatefulWidget {
  final int studentId;
  
  const CoursesPage({super.key, required this.studentId});

  @override
  State<CoursesPage> createState() => _CoursesPageState();
}

class _CoursesPageState extends State<CoursesPage> {
  final ApiClient _apiClient = ApiClient();
  
  bool _isLoading = true;
  String _errorMessage = '';
  List<Map<String, dynamic>> _courses = [];
  String _selectedFilter = 'Tous';

  @override
  void initState() {
    super.initState();
    _loadCourses();
  }

  Future<void> _loadCourses() async {
    try {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      final courses = await _apiClient.getCourses(widget.studentId);
      final prefs = await SharedPreferences.getInstance();
      
      setState(() {
        _courses = courses;

        // Fallback demo data if empty
        if (_courses.isEmpty) {
          _courses = [
            {
              'title': 'Physique Quantique',
              'professor': 'Prof. Sophie Laurent',
              'status': 'in_progress',
              'icon': 'science'
            },
            {
              'title': 'Algorithmique',
              'professor': 'Prof. Jean Dupont',
              'status': 'in_progress',
              'icon': 'computer'
            },
            {
              'title': 'Chimie Organique',
              'professor': 'Prof. Marie Bernard',
              'status': 'done',
              'icon': 'biotech'
            },
            {
              'title': 'Littérature Française',
              'professor': 'Prof. Pierre Moreau',
              'status': 'in_progress',
              'icon': 'book'
            },
            {
              'title': 'Mathématiques Avancées',
              'professor': 'Prof. Martin Dubois',
              'status': 'in_progress',
              'icon': 'math'
            }
          ];
        }

        // Apply local completion overrides
        for (var course in _courses) {
           final title = course['title'];
           final isLocallyDone = prefs.getBool('course_completed_$title');
           if (isLocallyDone == true) {
             course['status'] = 'done';
           } else if (isLocallyDone == false) {
             course['status'] = 'in_progress';
           }
        }

        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Erreur de chargement: ${e.toString()}';
      });
    }
  }

  IconData _getIcon(String? icon) {
    switch (icon?.toLowerCase()) {
      case 'math':
        return Icons.straighten;
      case 'science':
        return Icons.science_outlined;
      case 'computer':
        return Icons.computer_outlined;
      case 'biotech':
        return Icons.biotech_outlined;
      case 'book':
      default:
        return Icons.menu_book_outlined;
    }
  }

  Color _getIconBg(String? icon) {
    switch (icon?.toLowerCase()) {
      case 'math':
        return const Color(0xFFF9E8ED);
      case 'science':
        return const Color(0xFFEFF6FF);
      case 'computer':
        return const Color(0xFFE8F7F3);
      case 'biotech':
        return const Color(0xFFF2F5F7);
      case 'book':
      default:
        return const Color(0xFFFDF3D8);
    }
  }

  final TextEditingController _searchController = TextEditingController();

  List<Map<String, dynamic>> _getFilteredCourses() {
    // 1. Filter by tabs (Tous/En cours/Terminé)
    var list = _courses;
    if (_selectedFilter != 'Tous') {
      final filterStatus = _selectedFilter == 'En cours' ? 'in_progress' : 'done';
      list = _courses.where((c) => c['status'] == filterStatus).toList();
    }
    
    // 2. Filter by Search Query
    final query = _searchController.text.toLowerCase().trim();
    if (query.isNotEmpty) {
      list = list.where((c) {
        final title = c['title']?.toString().toLowerCase() ?? '';
        final professor = c['professor']?.toString().toLowerCase() ?? '';
        return title.contains(query) || professor.contains(query);
      }).toList();
    }
    
    return list;
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        backgroundColor: const Color(0xFFF6F6F6),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              CircularProgressIndicator(color: Color(0xFF2F65D9)),
              SizedBox(height: 16),
              Text('Chargement des cours...'),
            ],
          ),
        ),
      );
    }

    if (_errorMessage.isNotEmpty) {
      return Scaffold(
        backgroundColor: const Color(0xFFF6F6F6),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.error_outline, size: 64, color: Colors.red),
                const SizedBox(height: 16),
                Text(_errorMessage, textAlign: TextAlign.center),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: _loadCourses,
                  child: const Text('Réessayer'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    final filteredCourses = _getFilteredCourses();

    return Scaffold(
      backgroundColor: const Color(0xFFF6F6F6),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              child: Text(
                'Mes Cours',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF1F2937),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 2),
                child: TextField(
                  controller: _searchController,
                  onChanged: (value) => setState(() {}),
                  decoration: const InputDecoration(
                    hintText: 'Rechercher un cours...',
                    prefixIcon: Icon(Icons.search),
                    border: InputBorder.none,
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
              child: Row(
                children: [
                  _FilterChip(
                    label: 'Tous',
                    selected: _selectedFilter == 'Tous',
                    onTap: () => setState(() => _selectedFilter = 'Tous'),
                  ),
                  const SizedBox(width: 8),
                  _FilterChip(
                    label: 'En cours',
                    selected: _selectedFilter == 'En cours',
                    onTap: () => setState(() => _selectedFilter = 'En cours'),
                  ),
                  const SizedBox(width: 8),
                  _FilterChip(
                    label: 'Terminé',
                    selected: _selectedFilter == 'Terminé',
                    onTap: () => setState(() => _selectedFilter = 'Terminé'),
                  ),
                ],
              ),
            ),
            Expanded(
              child: filteredCourses.isEmpty
                  ? const Center(
                      child: Text(
                        'Aucun cours trouvé',
                        style: TextStyle(color: Colors.black54),
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      itemCount: filteredCourses.length,
                      itemBuilder: (context, index) {
                        final course = filteredCourses[index];
                        final isDone = course['status'] == 'done';
                        final title = course['title'] ?? 'Cours';
                        final professor = course['professor'] ?? 'Professeur';
                        final icon = _getIcon(course['icon']);
                        final iconBg = _getIconBg(course['icon']);

                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 6),
                          child: Card(
                            child: ListTile(
                              contentPadding: const EdgeInsets.all(14),
                              leading: Container(
                                width: 52,
                                height: 52,
                                decoration: BoxDecoration(
                                  color: iconBg,
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: Icon(icon, color: Colors.black54),
                              ),
                              title: Text(
                                title,
                                style: const TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              subtitle: Padding(
                                padding: const EdgeInsets.only(top: 4),
                                child: Row(
                                  children: [
                                    const Icon(Icons.person_outline,
                                        size: 16, color: Color(0xFF6B7280)),
                                    const SizedBox(width: 4),
                                    Text(
                                      professor,
                                      style: const TextStyle(
                                        color: Color(0xFF6B7280),
                                        fontSize: 13,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              trailing: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 12, vertical: 6),
                                decoration: BoxDecoration(
                                  color: isDone
                                      ? const Color(0xFFF1F2F6)
                                      : const Color(0xFFDFF9EB),
                                  borderRadius: BorderRadius.circular(18),
                                ),
                                child: Text(
                                  isDone ? 'Terminé' : 'En cours',
                                  style: TextStyle(
                                    color: isDone
                                        ? const Color(0xFF6B7280)
                                        : const Color(0xFF22C55E),
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                              onTap: () async {
                                final result = await Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (_) =>
                                        CourseDetailPage(courseTitle: title),
                                  ),
                                );
                                // If returned true, reload courses to update status
                                if (result == true) {
                                  _loadCourses();
                                }
                              },
                            ),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _FilterChip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFF2F65D9) : const Color(0xFFF1F2F6),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: selected ? Colors.white : const Color(0xFF6B7280),
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
    );
  }
}
