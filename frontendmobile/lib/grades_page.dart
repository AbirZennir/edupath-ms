import 'package:flutter/material.dart';

class GradesPage extends StatelessWidget {
  const GradesPage({super.key});

  @override
  Widget build(BuildContext context) {
    final grades = [
      ('Mathématiques', 15.5),
      ('Physique', 14.0),
      ('Informatique', 17.0),
      ('Chimie', 13.5),
    ];

    final average = grades.fold<double>(
            0, (sum, g) => sum + g.$2) /
        grades.length;

    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        title: const Text('Mes Notes'),
        backgroundColor: const Color(0xFFF5F7FB),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Card(
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: ListTile(
                title: const Text('Moyenne générale'),
                subtitle: const Text('Toutes matières confondues'),
                trailing: Text(
                  average.toStringAsFixed(2),
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF2563EB),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.builder(
                itemCount: grades.length,
                itemBuilder: (context, index) {
                  final g = grades[index];
                  return Card(
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: ListTile(
                      title: Text(g.$1),
                      trailing: Text(
                        g.$2.toStringAsFixed(2),
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 18,
                        ),
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
