import 'package:flutter/material.dart';
import 'course_detail_page.dart';

class CoursesPage extends StatelessWidget {
  const CoursesPage({super.key});

  @override
  Widget build(BuildContext context) {
    final courses = [
      ('Mathématiques Avancées', 'Prof. Martin Dubois', 'En cours',
          Icons.straighten, const Color(0xFFF9E8ED)),
      ('Physique Quantique', 'Prof. Sophie Laurent', 'En cours',
          Icons.science_outlined, const Color(0xFFEFF6FF)),
      ('Algorithmique', 'Prof. Jean Dupont', 'En cours',
          Icons.computer_outlined, const Color(0xFFE8F7F3)),
      ('Chimie Organique', 'Prof. Marie Bernard', 'Terminé',
          Icons.biotech_outlined, const Color(0xFFF2F5F7)),
      ('Littérature Française', 'Prof. Pierre Moreau', 'En cours',
          Icons.menu_book_outlined, const Color(0xFFFDF3D8)),
    ];

    return Scaffold(
      backgroundColor: const Color(0xFFF6F6F6),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              child: const Text(
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
                padding:
                    const EdgeInsets.symmetric(horizontal: 14, vertical: 2),
                child: const TextField(
                  decoration: InputDecoration(
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
                children: const [
                  _FilterChip(label: 'Tous', selected: true),
                  SizedBox(width: 8),
                  _FilterChip(label: 'En cours', selected: false),
                  SizedBox(width: 8),
                  _FilterChip(label: 'Terminé', selected: false),
                ],
              ),
            ),
            Expanded(
              child: ListView.builder(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                itemCount: courses.length,
                itemBuilder: (context, index) {
                  final c = courses[index];
                  final isDone = c.$3 == 'Terminé';
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6),
                    child: Card(
                      child: ListTile(
                        contentPadding: const EdgeInsets.all(14),
                        leading: Container(
                          width: 52,
                          height: 52,
                          decoration: BoxDecoration(
                            color: c.$5,
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: Icon(c.$4, color: Colors.black54),
                        ),
                        title: Text(
                          c.$1,
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
                                c.$2,
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
                            c.$3,
                            style: TextStyle(
                              color: isDone
                                  ? const Color(0xFF6B7280)
                                  : const Color(0xFF22C55E),
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) =>
                                  CourseDetailPage(courseTitle: c.$1),
                            ),
                          );
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

  const _FilterChip({required this.label, required this.selected});

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
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
    );
  }
}
