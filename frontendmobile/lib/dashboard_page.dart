import 'package:flutter/material.dart';

class DashboardPage extends StatelessWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FB),
      body: SafeArea(
        child: Column(
          children: [
            // Bandeau bleu
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              decoration: const BoxDecoration(
                color: Color(0xFF2563EB),
                borderRadius: BorderRadius.vertical(
                  bottom: Radius.circular(24),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const CircleAvatar(
                        backgroundImage: AssetImage('assets/avatar.png'),
                        radius: 20,
                      ),
                      const SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Bonjour, Sophie 👋',
                            style: theme.textTheme.titleMedium?.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            'Voici votre progression du jour',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: Colors.white70,
                            ),
                          ),
                        ],
                      )
                    ],
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Progression générale',
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: Colors.white70),
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(999),
                    child: LinearProgressIndicator(
                      value: 0.73,
                      minHeight: 10,
                      backgroundColor: Colors.white24,
                      valueColor:
                          const AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '+5% cette semaine',
                    style: theme.textTheme.bodySmall
                        ?.copyWith(color: Colors.white),
                  )
                ],
              ),
            ),

            // Contenu scrollable
            Expanded(
              child: ListView(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                children: [
                  // Cours en cours
                  _sectionHeader(
                    context,
                    title: 'Cours en cours',
                    action: 'Voir tout',
                  ),
                  const SizedBox(height: 8),
                  _CourseProgressCard(
                    title: 'Mathématiques',
                    percent: 0.75,
                    color: Colors.redAccent,
                  ),
                  const SizedBox(height: 8),
                  _CourseProgressCard(
                    title: 'Physique',
                    percent: 0.60,
                    color: Colors.blueAccent,
                  ),
                  const SizedBox(height: 8),
                  _CourseProgressCard(
                    title: 'Informatique',
                    percent: 0.85,
                    color: Colors.greenAccent,
                  ),
                  const SizedBox(height: 24),

                  // Devoirs urgents
                  _sectionHeader(
                    context,
                    title: 'Devoirs urgents',
                    action: 'Voir tout',
                  ),
                  const SizedBox(height: 8),
                  const _UrgentTaskCard(
                    title: 'Devoir Mathématiques Ch.5',
                    subtitle: 'À rendre bientôt',
                    daysLabel: '2 jours',
                    color: Color(0xFFF97373),
                  ),
                  const SizedBox(height: 8),
                  const _UrgentTaskCard(
                    title: 'TP Physique Quantique',
                    subtitle: 'À rendre bientôt',
                    daysLabel: '5 jours',
                    color: Color(0xFFFACC15),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _sectionHeader(BuildContext context,
      {required String title, String? action}) {
    final theme = Theme.of(context);
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          title,
          style: theme.textTheme.titleMedium
              ?.copyWith(fontWeight: FontWeight.w600),
        ),
        if (action != null)
          Text(
            action,
            style: theme.textTheme.bodySmall?.copyWith(
              color: const Color(0xFF2563EB),
              fontWeight: FontWeight.w500,
            ),
          ),
      ],
    );
  }
}

class _CourseProgressCard extends StatelessWidget {
  final String title;
  final double percent;
  final Color color;

  const _CourseProgressCard({
    required this.title,
    required this.percent,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: theme.textTheme.bodyMedium),
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                value: percent,
                minHeight: 8,
                backgroundColor: const Color(0xFFE5E7EB),
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            ),
            const SizedBox(height: 4),
            Align(
              alignment: Alignment.centerRight,
              child: Text('${(percent * 100).round()}%'),
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
  final String daysLabel;
  final Color color;

  const _UrgentTaskCard({
    required this.title,
    required this.subtitle,
    required this.daysLabel,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        leading: const Icon(Icons.description_outlined),
        title: Text(title,
            style:
                theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500)),
        subtitle: Text(subtitle),
        trailing: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(999),
          ),
          child: Text(
            daysLabel,
            style: TextStyle(color: color, fontWeight: FontWeight.w500),
          ),
        ),
      ),
    );
  }
}
