import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import '../api_client.dart';

class DashboardPage extends StatefulWidget {
  final int studentId;
  final Function(int) onTabChange;
  
  const DashboardPage({
    super.key, 
    required this.studentId,
    required this.onTabChange,
  });

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  final ApiClient _apiClient = ApiClient();
  
  bool _isLoading = true;
  String _errorMessage = '';
  
  // Dashboard data
  String _studentName = '';
  int _globalProgress = 0;
  int _weekChange = 0;
  List<Map<String, dynamic>> _courses = [];
  List<Map<String, dynamic>> _urgentAssignments = [];
  List<Map<String, dynamic>> _recommendations = [];

  @override
  void initState() {
    super.initState();
    _loadDashboardData();
  }

  Future<void> _loadDashboardData() async {
    try {
      if (!mounted) return;
      
      final data = await _apiClient.getDashboard(widget.studentId);
      final prefs = await SharedPreferences.getInstance();
      
      if (!mounted) return;

      setState(() {
        // Parse student info
        final student = data['student'] as Map<String, dynamic>?;
        _studentName = '${student?['prenom'] ?? ''} ${student?['nom'] ?? ''}'.trim();
        if (_studentName.isEmpty) _studentName = 'Étudiant';

        // Parse progression
        final progression = data['progression'] as Map<String, dynamic>?;
        _globalProgress = progression?['global'] ?? 0;
        _weekChange = progression?['weekChange'] ?? 0;

        // Parse courses
        List<Map<String, dynamic>> loadedCourses = (data['courses'] as List?)
                ?.map((c) => c as Map<String, dynamic>)
                .toList() ?? [];

        // FALLBACK FOR DEMO if empty
        if (loadedCourses.isEmpty) {
          loadedCourses = [
            {
              'title': 'Physique Quantique',
              'progress': 45,
              'icon': 'science'
            },
            {
              'title': 'Algorithmique',
              'progress': 70,
              'icon': 'computer'
            },
            {
              'title': 'Chimie Organique',
              'progress': 20,
              'icon': 'biotech'
            },
            {
              'title': 'Littérature Française',
              'progress': 10,
              'icon': 'book'
            },
             {
              'title': 'Mathématiques Avancées',
              'progress': 85,
              'icon': 'math'
            }
          ];
        }

        // Filter out locally completed courses from "Cours en cours" list
        _courses = loadedCourses.where((c) {
          final title = c['title'];
          final isDone = prefs.getBool('course_completed_$title') ?? false;
          // If it's done, we might want to exclude it from "In progress" or show it as 100%
          // User asked: "quand je termine qlq cours sa reste tjr en cours ? fix this pls"
          // So we should probably remove it or mark it 100% done.
          // Let's remove it from this specific list if we treat this section as "En cours" only.
          return !isDone;
        }).toList();

        // Update Global Progress based on done courses (Mock logic to make it dynamic)
        final totalCourses = loadedCourses.length;
        final completedCourses = loadedCourses.where((c) {
           final title = c['title'];
           return prefs.getBool('course_completed_$title') ?? false;
        }).length;
        
        if (totalCourses > 0) {
           // Simple mock math: existing progress + (completed count contribution)
           // or just recalculate based on completion.
           // Let's say each completed course is 100%, others are their current `%`.
           // But here we just want to update the general indicator slightly.
           if (completedCourses > 0) {
             // Mock update for visual feedback
             _globalProgress = ((completedCourses / totalCourses) * 100).round();
             if (_globalProgress < 15) _globalProgress = 15; // min
           }
        }

        // Parse urgent assignments
        _urgentAssignments = (data['urgentAssignments'] as List?)
                ?.map((a) => a as Map<String, dynamic>)
                .toList() ?? [];
        
        // FALLBACK FOR DEMO if empty
        if (_urgentAssignments.isEmpty) {
          _urgentAssignments = [
            {
              'title': 'Devoir Chapitre 3',
              'daysLeft': 2,
            },
             {
              'title': 'Exercices Matrices',
              'daysLeft': 7,
            }
          ];
        }



        _isLoading = false;
        
        // Trigger local AI analysis
        _generateSmartRecommendations();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _errorMessage = 'Erreur de chargement: ${e.toString()}';
      });
    }
  }

  void _generateSmartRecommendations() {
    final List<Map<String, dynamic>> aiSuggestions = [];

    // 1. Analyze Academic Weaknesses (Lowest Progress)
    if (_courses.isNotEmpty) {
      // Sort by progress ascending
      final sortedCourses = List<Map<String, dynamic>>.from(_courses);
      sortedCourses.sort((a, b) => (a['progress'] as int).compareTo(b['progress'] as int));
      
      final weakCourse = sortedCourses.first;
      final weakTitle = weakCourse['title'];
      final progress = weakCourse['progress'];

      if (progress < 50) {
        aiSuggestions.add({
          'type': 'video',
          'title': 'Rattrapage : $weakTitle',
          'reason': 'Progression faible ($progress%) détectée',
          'duration': '20 min',
          'color': const Color(0xFFFFEDD5),
          'textColor': const Color(0xFFC2410C),
          'url': _getVideoUrl(weakTitle),
        });
      }
    }

    // 2. Analyze Urgency (Assignments due soon)
    final urgent = _urgentAssignments.where((a) => (a['daysLeft'] ?? 99) <= 2).toList();
    if (urgent.isNotEmpty) {
      aiSuggestions.add({
        'type': 'exercise',
        'title': 'Session Focus : ${urgent.first['title']}',
        'reason': 'Urgent : À rendre dans ${urgent.first['daysLeft']} jours',
        'duration': '45 min',
        'color': const Color(0xFFFDE2E1),
        'textColor': const Color(0xFFE43B3B),
        'url': null, // Could link to a timer/focus mode
      });
    }

    // 3. Analyze Strengths (High Progress)
    final strongCourse = _courses.firstWhere(
      (c) => (c['progress'] as int) > 80, 
      orElse: () => {},
    );
    
    if (strongCourse.isNotEmpty) {
       aiSuggestions.add({
        'type': 'tutor',
        'title': 'Aller plus loin en ${strongCourse['title']}',
        'reason': 'Excellente maîtrise ! Approfondissez.',
        'duration': 'Article',
        'color': const Color(0xFFE0F2FE),
        'textColor': const Color(0xFF0369A1),
        'url': _getAdvancedUrl(strongCourse['title']),
      });
    }

    // 4. General fallback if list is short
    if (aiSuggestions.length < 2) {
      aiSuggestions.add({
         'type': 'tutor', 
         'title': 'Planification Hebdomadaire',
         'reason': 'Optimisez votre temps d\'étude',
         'duration': '10 min',
         'color': const Color(0xFFF3E8FF),
         'textColor': const Color(0xFF7E22CE),
         'url': 'https://calendar.google.com/',
      });
    }

    _recommendations = aiSuggestions;
  }

  String _getVideoUrl(String? course) {
    switch (course) {
      case 'Physique Quantique': return 'https://www.youtube.com/watch?v=3-Ps97M_BaY';
      case 'Algorithmique': return 'https://www.youtube.com/watch?v=m_yj2p6SCEM';
      case 'Chimie Organique': return 'https://www.youtube.com/watch?v=ZCFEqsxlfQc';
      default: return 'https://www.youtube.com/results?search_query=cours+${Uri.encodeComponent(course ?? "")}';
    }
  }

  String _getAdvancedUrl(String? course) {
    // Links to advanced papers or wikipedia
    return 'https://scholar.google.com/scholar?q=${Uri.encodeComponent(course ?? "")}';
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
              Text('Chargement...', style: TextStyle(color: Colors.black54)),
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
                Text(
                  _errorMessage,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.black87),
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: _loadDashboardData,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF2F65D9),
                  ),
                  child: const Text('Réessayer'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFFF6F6F6),
      body: SafeArea(
        child: Stack(
          children: [
            Container(
              height: 260,
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [Color(0xFF2F65D9), Color(0xFF4F7DFF)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.vertical(
                  bottom: Radius.circular(32),
                ),
              ),
            ),
            ListView(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              children: [
                _header(),
                const SizedBox(height: 16),
                _progressCard(),
                const SizedBox(height: 20),
                _section(
                  title: 'Cours en cours',
                  action: 'Voir tout',
                  onAction: () => widget.onTabChange(1), // Go to Courses tab
                  child: _courses.isEmpty
                      ? const Padding(
                          padding: EdgeInsets.all(16.0),
                          child: Text(
                            'Aucun cours en cours (Tout est terminé ! 🎉)',
                            style: TextStyle(color: Colors.black54),
                          ),
                        )
                      : Column(
                          children: _courses
                              .map((course) => Padding(
                                    padding: const EdgeInsets.only(bottom: 12),
                                    child: _CourseProgressCard(
                                      title: course['title'] ?? 'Cours',
                                      percent: (course['progress'] ?? 0) / 100.0,
                                      color: const Color(0xFF25C5C9),
                                      icon: _getIconFromString(course['icon']),
                                      iconBg: const Color(0xFFF9E8ED),
                                    ),
                                  ))
                              .toList(),
                        ),
                ),
                const SizedBox(height: 20),
                _section(
                  title: 'Devoirs urgents',
                  action: 'Voir tout',
                  onAction: () => widget.onTabChange(2), // Go to Assignments tab
                  child: _urgentAssignments.isEmpty
                      ? const Padding(
                          padding: EdgeInsets.all(16.0),
                          child: Text(
                            'Aucun devoir urgent',
                            style: TextStyle(color: Colors.black54),
                          ),
                        )
                      : Column(
                          children: _urgentAssignments
                              .map((assignment) => Padding(
                                    padding: const EdgeInsets.only(bottom: 12),
                                    child: _UrgentTaskCard(
                                      title: assignment['title'] ?? 'Devoir',
                                      subtitle: 'À rendre bientôt',
                                      badge: '${assignment['daysLeft'] ?? 0} jours',
                                      badgeColor: const Color(0xFFFDE2E1),
                                      badgeTextColor: const Color(0xFFE43B3B),
                                      icon: Icons.description_outlined,
                                      iconBg: const Color(0xFFF9E8ED),
                                    ),
                                  ))
                              .toList(),
                        ),
                ),
                const SizedBox(height: 20),
                _section(
                  title: 'Recommandations Ciblées (IA)',
                  action: 'Tout voir',
                  onAction: () {}, 
                  child: SizedBox(
                    height: 180,
                    child: ListView.separated(
                      scrollDirection: Axis.horizontal,
                      itemCount: _recommendations.length,
                      separatorBuilder: (context, index) => const SizedBox(width: 12),
                      itemBuilder: (context, index) {
                        return _RecommendationCard(item: _recommendations[index]);
                      },
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  IconData _getIconFromString(String? iconName) {
    switch (iconName?.toLowerCase()) {
      case 'math':
        return Icons.straighten;
      case 'computer':
        return Icons.computer_outlined;
      case 'science':
        return Icons.science_outlined;
      case 'biotech':
        return Icons.biotech_outlined;
      default:
        return Icons.menu_book_outlined;
    }
  }

  Widget _header() {
    return Row(
      children: [
        const CircleAvatar(
          radius: 24,
          backgroundColor: Colors.white24,
          child: Icon(Icons.person_outline, color: Colors.white, size: 26),
        ),
        const SizedBox(width: 12),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Bonjour, $_studentName 👋',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 4),
            const Text(
              'Voici votre progression du jour',
              style: TextStyle(
                color: Colors.white70,
                fontSize: 13,
              ),
            ),
          ],
        )
      ],
    );
  }

  Widget _progressCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.12),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Progression générale',
                style: TextStyle(color: Colors.white, fontSize: 14),
              ),
              Text(
                '$_globalProgress%',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              value: _globalProgress / 100.0,
              minHeight: 10,
              backgroundColor: const Color(0xFF6F8FE8),
              valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
            ),
          ),
          const SizedBox(height: 10),
          Text(
            '↗ +$_weekChange% cette semaine',
            style: const TextStyle(color: Colors.white, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _section(
      {required String title,
      required String action,
      required Widget child,
      VoidCallback? onAction}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: Color(0xFF1F2937),
              ),
            ),
            GestureDetector(
              onTap: onAction,
              child: Text(
                action,
                style: const TextStyle(
                  color: Color(0xFF2F65D9),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        child,
      ],
    );
  }
}

class _CourseProgressCard extends StatelessWidget {
  final String title;
  final double percent;
  final Color color;
  final IconData icon;
  final Color iconBg;

  const _CourseProgressCard({
    required this.title,
    required this.percent,
    required this.color,
    required this.icon,
    required this.iconBg,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: iconBg,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(icon, color: Colors.black54),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                value: percent,
                minHeight: 8,
                backgroundColor: const Color(0xFFE5E7EB),
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            ),
            const SizedBox(height: 6),
            Align(
              alignment: Alignment.centerRight,
              child: Text(
                '${(percent * 100).round()}%',
                style: const TextStyle(
                  color: Color(0xFF6B7280),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _UrgentTaskCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final String badge;
  final Color badgeColor;
  final Color badgeTextColor;
  final IconData icon;
  final Color iconBg;

  const _UrgentTaskCard({
    required this.title,
    required this.subtitle,
    required this.badge,
    required this.badgeColor,
    required this.badgeTextColor,
    required this.icon,
    required this.iconBg,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: iconBg,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: Colors.black54),
        ),
        title: Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 15,
          ),
        ),
        subtitle: Row(
          children: const [
            Icon(Icons.access_time, size: 14, color: Color(0xFF6B7280)),
            SizedBox(width: 4),
            Text('À rendre bientôt',
                style: TextStyle(color: Color(0xFF6B7280), fontSize: 12)),
          ],
        ),
        trailing: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: badgeColor,
            borderRadius: BorderRadius.circular(999),
          ),
          child: Text(
            badge,
            style: TextStyle(
              color: badgeTextColor,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ),
    );
  }
}

class _RecommendationCard extends StatelessWidget {
  final Map<String, dynamic> item;

  const _RecommendationCard({required this.item});

  IconData _getIcon() {
    switch (item['type']) {
      case 'video': return Icons.play_circle_fill;
      case 'exercise': return Icons.quiz;
      case 'tutor': return Icons.school;
      default: return Icons.star;
    }
  }

  String _getLabel() {
    switch (item['type']) {
      case 'video': return 'Vidéo recommandée';
      case 'exercise': return 'Exercice ciblé';
      case 'tutor': return 'Soutien';
      default: return 'Conseil';
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () async {
        final url = item['url'];
        if (url != null) {
          final uri = Uri.parse(url);
          if (await canLaunchUrl(uri)) {
             await launchUrl(uri, mode: LaunchMode.externalApplication);
          }
        }
      },
      child: Container(
        width: 240,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: item['color'],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(_getIcon(), size: 14, color: item['textColor']),
                  const SizedBox(width: 6),
                  Text(
                    _getLabel(),
                    style: TextStyle(
                      color: item['textColor'],
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            Text(
              item['title'],
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontWeight: FontWeight.w700,
                fontSize: 15,
                height: 1.3,
              ),
            ),
            const Spacer(),
            Row(
              children: [
                const Icon(Icons.auto_awesome, size: 14, color: Color(0xFF6B7280)),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    item['reason'],
                    style: const TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 11,
                      fontStyle: FontStyle.italic,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
             Row(
              children: [
                Icon(Icons.timer_outlined, size: 14, color: Colors.grey[400]),
                const SizedBox(width: 4),
                Text(
                  item['duration'],
                  style: TextStyle(color: Colors.grey[400], fontSize: 12),
                ),
                const Spacer(),
                const Icon(Icons.arrow_forward_rounded, size: 18, color: Color(0xFF2F65D9)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
