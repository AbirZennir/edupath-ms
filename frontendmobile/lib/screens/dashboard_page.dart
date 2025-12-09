import 'package:flutter/material.dart';

class DashboardPage extends StatelessWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context) {
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
                  child: Column(
                    children: const [
                      _CourseProgressCard(
                        title: 'Mathématiques',
                        percent: 0.60,
                        color: Color(0xFF25C5C9),
                        icon: Icons.straighten,
                        iconBg: Color(0xFFF9E8ED),
                      ),
                      SizedBox(height: 12),
                      _CourseProgressCard(
                        title: 'Informatique',
                        percent: 0.85,
                        color: Color(0xFF25C5C9),
                        icon: Icons.laptop_mac,
                        iconBg: Color(0xFFE8F7F3),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                _section(
                  title: 'Devoirs urgents',
                  action: 'Voir tout',
                  child: Column(
                    children: const [
                      _UrgentTaskCard(
                        title: 'Devoir Mathématiques Ch.5',
                        subtitle: 'À rendre bientôt',
                        badge: '2 jours',
                        badgeColor: Color(0xFFFDE2E1),
                        badgeTextColor: Color(0xFFE43B3B),
                        icon: Icons.description_outlined,
                        iconBg: Color(0xFFF9E8ED),
                      ),
                      SizedBox(height: 12),
                      _UrgentTaskCard(
                        title: 'TP Physique Quantique',
                        subtitle: 'À rendre bientôt',
                        badge: '5 jours',
                        badgeColor: Color(0xFFFDF3D8),
                        badgeTextColor: Color(0xFFE9B400),
                        icon: Icons.science_outlined,
                        iconBg: Color(0xFFEFF6FF),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
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
          children: const [
            Text(
              'Bonjour, Sophie 👋',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
            SizedBox(height: 4),
            Text(
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
            children: const [
              Text(
                'Progression générale',
                style: TextStyle(color: Colors.white, fontSize: 14),
              ),
              Text(
                '73%',
                style: TextStyle(
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
            child: const LinearProgressIndicator(
              value: 0.73,
              minHeight: 10,
              backgroundColor: Color(0xFF6F8FE8),
              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            '↗ +5% cette semaine',
            style: TextStyle(color: Colors.white, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _section(
      {required String title, required String action, required Widget child}) {
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
            Text(
              action,
              style: const TextStyle(
                color: Color(0xFF2F65D9),
                fontWeight: FontWeight.w600,
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
