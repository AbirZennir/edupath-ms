import 'package:flutter/material.dart';

class CourseDetailPage extends StatelessWidget {
  final String courseTitle;

  const CourseDetailPage({super.key, required this.courseTitle});

  @override
  Widget build(BuildContext context) {
    final chapters = [
      'Introduction aux matrices',
      'Déterminants et systèmes linéaires',
      'Espaces vectoriels',
      'Applications linéaires',
      'Diagonalisation',
    ];

    final assignments = [
      'Devoir Chapitre 3',
      'Exercices Matrices',
    ];

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // header coloré
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              color: Colors.redAccent,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    courseTitle,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Prof. Martin Dubois',
                    style: TextStyle(color: Colors.white70),
                  ),
                ],
              ),
            ),

            Expanded(
              child: ListView(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                children: [
                  const Text(
                    'Description du cours',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Ce cours couvre des concepts avancés des mathématiques incluant l’algèbre linéaire, le calcul différentiel et intégral, ainsi que les équations différentielles.',
                  ),
                  const SizedBox(height: 24),

                  const Text(
                    'Chapitres',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...chapters.asMap().entries.map(
                        (e) => Card(
                          elevation: 0,
                          color: const Color(0xFFEFFDF3),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: ListTile(
                            leading: CircleAvatar(
                              radius: 14,
                              backgroundColor: const Color(0xFF22C55E),
                              child: Text(
                                '${e.key + 1}',
                                style: const TextStyle(
                                    color: Colors.white, fontSize: 12),
                              ),
                            ),
                            title: Text(e.value),
                            trailing: const Icon(
                              Icons.check,
                              color: Color(0xFF22C55E),
                            ),
                          ),
                        ),
                      ),
                  const SizedBox(height: 24),

                  const Text(
                    'Devoirs du cours',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...assignments.map(
                    (a) => Card(
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ListTile(
                        leading: const Icon(Icons.description_outlined),
                        title: Text(a),
                        subtitle: const Text('Mathématiques'),
                        trailing: const Text(
                          '1 semaine',
                          style: TextStyle(color: Colors.blueGrey),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: FilledButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.play_arrow),
                      label: const Text('Commencer le cours'),
                    ),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
