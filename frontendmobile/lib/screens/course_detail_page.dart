import 'package:flutter/material.dart';

class CourseDetailPage extends StatelessWidget {
  final String courseTitle;

  const CourseDetailPage({super.key, required this.courseTitle});

  @override
  Widget build(BuildContext context) {
    final chapters = [
      ('Introduction aux matrices', true),
      ('Déterminants et systèmes linéaires', true),
      ('Espaces vectoriels', true),
      ('Applications linéaires', false),
      ('Diagonalisation', false),
    ];

    final assignments = [
      ('Devoir Chapitre 3', '2 jours', const Color(0xFFFDE2E1),
          const Color(0xFFE43B3B)),
      ('Exercices Matrices', '1 semaine', const Color(0xFFE8F0FF),
          const Color(0xFF2F65D9)),
    ];

    return Scaffold(
      backgroundColor: const Color(0xFFF6F6F6),
      body: SafeArea(
        child: Column(
          children: [
            _header(context),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                children: [
                  _sectionTitle('Description du cours'),
                  _card(
                    child: const Text(
                      'Ce cours couvre les concepts avancés en mathématiques incluant l’algèbre linéaire, le calcul différentiel et intégral, ainsi que les équations différentielles.',
                      style: TextStyle(
                        color: Color(0xFF4B5563),
                        height: 1.4,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _sectionTitle('Chapitres'),
                  _card(
                    child: Column(
                      children: [
                        ...chapters.asMap().entries.map(
                          (entry) => _chapterTile(
                            index: entry.key + 1,
                            title: entry.value.$1,
                            done: entry.value.$2,
                            isLast: entry.key == chapters.length - 1,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  _sectionTitle('Devoirs du cours'),
                  _card(
                    child: Column(
                      children: assignments
                          .map(
                            (a) => _assignmentTile(
                              title: a.$1,
                              deadline: a.$2,
                              badgeColor: a.$3,
                              badgeTextColor: a.$4,
                            ),
                          )
                          .toList(),
                    ),
                  ),
                  const SizedBox(height: 18),
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF2F65D9),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      onPressed: () {},
                      icon: const Icon(Icons.play_arrow),
                      label: const Text(
                        'Commencer le cours',
                        style: TextStyle(fontWeight: FontWeight.w700),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _header(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 12, 8, 18),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFFF16068), Color(0xFFF2777F)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.vertical(
          bottom: Radius.circular(24),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.arrow_back, color: Colors.white),
          ),
          const SizedBox(width: 8),
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(14),
            ),
            child:
                const Icon(Icons.straighten, color: Colors.white, size: 26),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                courseTitle,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 4),
              const Text(
                'Prof. Martin Dubois',
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 13,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        text,
        style: const TextStyle(
          fontWeight: FontWeight.w700,
          fontSize: 15,
          color: Color(0xFF1F2937),
        ),
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: child,
      ),
    );
  }

  Widget _chapterTile({
    required int index,
    required String title,
    required bool done,
    required bool isLast,
  }) {
    return Column(
      children: [
        Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: done
                    ? const Color(0xFFDFF9EB)
                    : const Color(0xFFF1F2F6),
                borderRadius: BorderRadius.circular(10),
              ),
              alignment: Alignment.center,
              child: Text(
                '$index',
                style: TextStyle(
                  color: done
                      ? const Color(0xFF22C55E)
                      : const Color(0xFF6B7280),
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                ),
              ),
            ),
            Icon(
              done ? Icons.check : Icons.more_horiz,
              color: done ? const Color(0xFF22C55E) : const Color(0xFF9CA3AF),
            ),
          ],
        ),
        if (!isLast)
          const Divider(
            height: 16,
            thickness: 1,
            color: Color(0xFFE5E7EB),
          ),
      ],
    );
  }

  Widget _assignmentTile({
    required String title,
    required String deadline,
    required Color badgeColor,
    required Color badgeTextColor,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                children: const [
                  Icon(Icons.access_time,
                      size: 14, color: Color(0xFF6B7280)),
                  SizedBox(width: 6),
                  Text(
                    'À rendre bientôt',
                    style: TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ],
          ),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: badgeColor,
              borderRadius: BorderRadius.circular(999),
            ),
            child: Text(
              deadline,
              style: TextStyle(
                color: badgeTextColor,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
