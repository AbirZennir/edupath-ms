import 'package:flutter/material.dart';

class AssignmentsPage extends StatelessWidget {
  const AssignmentsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          elevation: 0,
          title: const Text('Mes Devoirs'),
          backgroundColor: const Color(0xFFF5F7FB),
          bottom: const TabBar(
            indicatorSize: TabBarIndicatorSize.tab,
            indicatorWeight: 3,
            tabs: [
              Tab(text: 'Tous'),
              Tab(text: 'À rendre'),
              Tab(text: 'Rendus'),
            ],
          ),
        ),
        body: const TabBarView(
          children: [
            _AssignmentsList(type: 'all'),
            _AssignmentsList(type: 'todo'),
            _AssignmentsList(type: 'done'),
          ],
        ),
      ),
    );
  }
}

class _AssignmentsList extends StatelessWidget {
  final String type; // all / todo / done

  const _AssignmentsList({required this.type});

  @override
  Widget build(BuildContext context) {
    final all = [
      ('Devoir Mathématiques Ch.5', 'Mathématiques', 'Dans 2 jours', 'todo'),
      ('TP Physique Quantique', 'Physique', 'Dans 5 jours', 'todo'),
      ('Projet Algorithmique', 'Informatique', 'Dans 1 semaine', 'todo'),
      ('Exercices Algèbre', 'Mathématiques', 'Rendu il y a 3 jours', 'done'),
      ('Rapport de Chimie', 'Chimie', 'Rendu il y a 1 semaine', 'done'),
    ];

    final filtered = type == 'all'
        ? all
        : all.where((a) => a.$4 == type).toList();

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: filtered.length,
      itemBuilder: (context, index) {
        final a = filtered[index];
        final isDone = a.$4 == 'done';
        final color = isDone
            ? const Color(0xFF22C55E)
            : const Color(0xFFF97373);

        return Card(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: ListTile(
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            leading: const Icon(Icons.description_outlined),
            title: Text(a.$1),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(a.$2),
                const SizedBox(height: 4),
                Row(
                  children: [
                    const Icon(Icons.access_time, size: 14),
                    const SizedBox(width: 4),
                    Text(a.$3, style: const TextStyle(fontSize: 12)),
                  ],
                ),
              ],
            ),
            trailing: Icon(
              isDone ? Icons.check_circle : Icons.error_outline,
              color: color,
            ),
          ),
        );
      },
    );
  }
}
