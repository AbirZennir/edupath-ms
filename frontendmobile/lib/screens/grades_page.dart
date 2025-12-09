import 'package:flutter/material.dart';

class GradesPage extends StatelessWidget {
  const GradesPage({super.key});

  @override
  Widget build(BuildContext context) {
    final grades = [
      ('Mathématiques', 16.5, 'Excellent travail, continuez ainsi !',
          const Color(0xFFEB6A77)),
      ('Physique', 14.0, 'Bon effort, quelques points à améliorer.',
          const Color(0xFF2AC6A0)),
      ('Informatique', 18.0, 'Parfait ! Maîtrise exemplaire.',
          const Color(0xFF5EC6E8)),
      ('Chimie', 13.5, 'Satisfaisant, réviser le chapitre 3.',
          const Color(0xFF2AC6A0)),
      ('Littérature', 15.0, 'Bonne participation en cours.',
          const Color(0xFFE5BF3C)),
    ];

    final average =
        grades.fold<double>(0, (sum, g) => sum + g.$2) / grades.length;

    return Scaffold(
      backgroundColor: const Color(0xFFF6F6F6),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
          children: [
            const Text(
              'Mes Notes',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 14),
            _averageCard(average),
            const SizedBox(height: 16),
            ...grades.map(
              (g) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: _gradeTile(
                  subject: g.$1,
                  grade: g.$2,
                  comment: g.$3,
                  barColor: g.$4,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _averageCard(double average) {
    final double ratio = (average / 20).clamp(0.0, 1.0);
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF2F65D9), Color(0xFF4F7DFF)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Moyenne générale',
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '${average.toStringAsFixed(2)} / 20',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                  fontSize: 22,
                ),
              ),
              Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    width: 56,
                    height: 56,
                    child: CircularProgressIndicator(
                      value: ratio,
                      strokeWidth: 6,
                      backgroundColor: Colors.white24,
                      valueColor: const AlwaysStoppedAnimation<Color>(
                        Colors.white,
                      ),
                    ),
                  ),
                  Text(
                    '${(ratio * 100).round()}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                      fontSize: 13,
                    ),
                  ),
                ],
              )
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            '↘ -0.5 points ce mois-ci',
            style: TextStyle(color: Colors.white, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _gradeTile({
    required String subject,
    required double grade,
    required String comment,
    required Color barColor,
  }) {
    final double ratio = (grade / 20).clamp(0.0, 1.0);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  subject,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
                Text(
                  grade.toStringAsFixed(1),
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              comment,
              style: const TextStyle(
                color: Color(0xFF6B7280),
                fontSize: 12,
              ),
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                value: ratio,
                minHeight: 8,
                backgroundColor: const Color(0xFFE5E7EB),
                valueColor: AlwaysStoppedAnimation<Color>(barColor),
              ),
            ),
            const SizedBox(height: 4),
            const Align(
              alignment: Alignment.centerRight,
              child: Text(
                '/ 20',
                style: TextStyle(
                  color: Color(0xFF6B7280),
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
