import 'package:flutter/material.dart';
import 'course_detail_page.dart';

class CoursesPage extends StatelessWidget {
  const CoursesPage({super.key});

  @override
  Widget build(BuildContext context) {
    final courses = [
      ('Mathématiques Avancées', 'Prof. Martin Dubois', 'En cours'),
      ('Physique Quantique', 'Prof. Sophie Laurent', 'En cours'),
      ('Algorithmique', 'Prof. Jean Dupont', 'En cours'),
      ('Chimie Organique', 'Prof. Marie Bernard', 'Terminé'),
      ('Littérature Française', 'Prof. Pierre Moreau', 'En cours'),
    ];

    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        title: const Text('Mes Cours'),
        backgroundColor: const Color(0xFFF5F7FB),
      ),
      body: Column(
        children: [
          Padding(
            padding:
                const EdgeInsets.symmetric(horizontal: 16).copyWith(bottom: 8),
            child: TextField(
              decoration: const InputDecoration(
                hintText: 'Rechercher un cours…',
                prefixIcon: Icon(Icons.search),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                _filterChip('Tous', true),
                const SizedBox(width: 8),
                _filterChip('En cours', false),
                const SizedBox(width: 8),
                _filterChip('Terminé', false),
              ],
            ),
          ),
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              itemCount: courses.length,
              itemBuilder: (context, index) {
                final c = courses[index];
                final isDone = c.$3 == 'Terminé';
                return Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: ListTile(
                    contentPadding: const EdgeInsets.all(16),
                    leading: const CircleAvatar(
                      child: Icon(Icons.menu_book_outlined),
                    ),
                    title: Text(c.$1),
                    subtitle: Text(c.$2),
                    trailing: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                        color: (isDone
                                ? const Color(0xFF22C55E)
                                : const Color(0xFF2563EB))
                            .withOpacity(0.1),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Text(
                        c.$3,
                        style: TextStyle(
                          color: isDone
                              ? const Color(0xFF22C55E)
                              : const Color(0xFF2563EB),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => CourseDetailPage(courseTitle: c.$1),
                        ),
                      );
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _filterChip(String label, bool selected) {
    return ChoiceChip(
      label: Text(label),
      selected: selected,
      selectedColor: const Color(0xFF2563EB),
      labelStyle: TextStyle(
        color: selected ? Colors.white : Colors.black87,
      ),
    );
  }
}
