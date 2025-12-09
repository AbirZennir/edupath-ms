import 'package:flutter/material.dart';

class AssignmentsPage extends StatelessWidget {
  const AssignmentsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final assignments = [
      ('Devoir Mathématiques Ch.5', 'Mathématiques', 'Dans 2 jours', 'todo'),
      ('TP Physique Quantique', 'Physique', 'Dans 5 jours', 'todo'),
      ('Projet Algorithmique', 'Informatique', 'Dans 1 semaine', 'todo'),
      ('Exercices Algèbre', 'Mathématiques', 'Rendu il y a 3 jours', 'done'),
      ('Rapport de Chimie', 'Chimie', 'Rendu il y a 1 semaine', 'done'),
    ];

    return DefaultTabController(
      length: 3,
      child: Scaffold(
        backgroundColor: const Color(0xFFF6F6F6),
        body: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Padding(
                padding: EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: Text(
                  'Mes Devoirs',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: _tabBar(),
              ),
              Expanded(
                child: TabBarView(
                  children: [
                    _AssignmentsList(
                      data: assignments,
                      filter: 'all',
                    ),
                    _AssignmentsList(
                      data: assignments,
                      filter: 'todo',
                    ),
                    _AssignmentsList(
                      data: assignments,
                      filter: 'done',
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _tabBar() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFFF1F2F6),
        borderRadius: BorderRadius.circular(14),
      ),
      child: const TabBar(
        indicator: BoxDecoration(
          color: Color(0xFF2F65D9),
          borderRadius: BorderRadius.all(Radius.circular(12)),
        ),
        labelColor: Colors.white,
        unselectedLabelColor: Color(0xFF6B7280),
        dividerColor: Colors.transparent,
        tabs: [
          Tab(text: 'Tous'),
          Tab(text: 'À rendre'),
          Tab(text: 'Rendus'),
        ],
      ),
    );
  }
}

class _AssignmentsList extends StatelessWidget {
  final List<(String, String, String, String)> data;
  final String filter; // all | todo | done

  const _AssignmentsList({required this.data, required this.filter});

  @override
  Widget build(BuildContext context) {
    final filtered = filter == 'all'
        ? data
        : data.where((a) => a.$4 == filter).toList();

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      itemCount: filtered.length,
      itemBuilder: (context, index) {
        final item = filtered[index];
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: _AssignmentCard(
            title: item.$1,
            course: item.$2,
            due: item.$3,
            status: item.$4,
          ),
        );
      },
    );
  }
}

class _AssignmentCard extends StatelessWidget {
  final String title;
  final String course;
  final String due;
  final String status;

  const _AssignmentCard({
    required this.title,
    required this.course,
    required this.due,
    required this.status,
  });

  @override
  Widget build(BuildContext context) {
    final isDone = status == 'done';
    final color = isDone
        ? const Color(0xFF22C55E)
        : status == 'todo'
            ? const Color(0xFFF97373)
            : const Color(0xFF6B7280);
    final badgeBg = isDone
        ? const Color(0xFFDFF9EB)
        : status == 'todo'
            ? const Color(0xFFFDE2E1)
            : const Color(0xFFF1F2F6);

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
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: badgeBg,
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Icon(
                    isDone
                        ? Icons.check_circle
                        : status == 'todo'
                            ? Icons.error_outline
                            : Icons.info_outline,
                    color: color,
                    size: 18,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              course,
              style: const TextStyle(
                color: Color(0xFF6B7280),
                fontSize: 13,
              ),
            ),
            const SizedBox(height: 6),
            Row(
              children: [
                const Icon(Icons.access_time,
                    size: 14, color: Color(0xFF6B7280)),
                const SizedBox(width: 6),
                Text(
                  due,
                  style: const TextStyle(
                    color: Color(0xFF6B7280),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
