import 'package:flutter/material.dart';
import '../api_client.dart';

class AssignmentsPage extends StatefulWidget {
  final int studentId;
  
  const AssignmentsPage({super.key, required this.studentId});

  @override
  State<AssignmentsPage> createState() => _AssignmentsPageState();
}

class _AssignmentsPageState extends State<AssignmentsPage> {
  final ApiClient _apiClient = ApiClient();
  
  bool _isLoading = true;
  String _errorMessage = '';
  List<Map<String, dynamic>> _assignments = [];
  String _selectedFilter = 'Tous';

  @override
  void initState() {
    super.initState();
    _loadAssignments();
  }

  Future<void> _loadAssignments() async {
    try {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      final assignments = await _apiClient.getAssignments(widget.studentId);
      
      setState(() {
        _assignments = assignments;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Erreur: ${e.toString()}';
      });
    }
  }

  List<Map<String, dynamic>> _getFilteredAssignments() {
    if (_selectedFilter == 'Tous') return _assignments;
    
    final filterStatus = _selectedFilter == 'En attente' ? 'pending' : 'done';
    return _assignments.where((a) => a['status'] == filterStatus).toList();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        backgroundColor: const Color(0xFFF6F6F6),
        body: const Center(
          child: CircularProgressIndicator(color: Color(0xFF2F65D9)),
        ),
      );
    }

    if (_errorMessage.isNotEmpty) {
      return Scaffold(
        backgroundColor: const Color(0xFFF6F6F6),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(_errorMessage, textAlign: TextAlign.center),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _loadAssignments,
                child: const Text('Réessayer'),
              ),
            ],
          ),
        ),
      );
    }

    final filteredAssignments = _getFilteredAssignments();

    return Scaffold(
      backgroundColor: const Color(0xFFF6F6F6),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              child: Text(
                'Mes Devoirs',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF1F2937),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Row(
                children: [
                  _FilterChip(
                    label: 'Tous',
                    selected: _selectedFilter == 'Tous',
                    onTap: () => setState(() => _selectedFilter = 'Tous'),
                  ),
                  const SizedBox(width: 8),
                  _FilterChip(
                    label: 'En attente',
                    selected: _selectedFilter == 'En attente',
                    onTap: () => setState(() => _selectedFilter = 'En attente'),
                  ),
                  const SizedBox(width: 8),
                  _FilterChip(
                    label: 'Terminé',
                    selected: _selectedFilter == 'Terminé',
                    onTap: () => setState(() => _selectedFilter = 'Terminé'),
                  ),
                ],
              ),
            ),
            Expanded(
              child: filteredAssignments.isEmpty
                  ? const Center(
                      child: Text(
                        'Aucun devoir trouvé',
                        style: TextStyle(color: Colors.black54),
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      itemCount: filteredAssignments.length,
                      itemBuilder: (context, index) {
                        final assignment = filteredAssignments[index];
                        final title = assignment['title'] ?? 'Devoir';
                        final course = assignment['course'] ?? 'Cours';
                        final daysLeft = assignment['dueInDays'] ?? 0;
                        final status = assignment['status'] ?? 'pending';
                        final isDone = status == 'done';

                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 6),
                          child: Card(
                            child: ListTile(
                              contentPadding: const EdgeInsets.all(14),
                              leading: Container(
                                width: 52,
                                height: 52,
                                decoration: BoxDecoration(
                                  color: const Color(0xFFF9E8ED),
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: const Icon(
                                  Icons.description_outlined,
                                  color: Colors.black54,
                                ),
                              ),
                              title: Text(
                                title,
                                style: const TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              subtitle: Padding(
                                padding: const EdgeInsets.only(top: 4),
                                child: Text(
                                  course,
                                  style: const TextStyle(
                                    color: Color(0xFF6B7280),
                                    fontSize: 13,
                                  ),
                                ),
                              ),
                              trailing: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 12, vertical: 6),
                                decoration: BoxDecoration(
                                  color: isDone
                                      ? const Color(0xFFDFF9EB)
                                      : (daysLeft < 0
                                          ? const Color(0xFFFDE2E1)
                                          : const Color(0xFFFDF3D8)),
                                  borderRadius: BorderRadius.circular(18),
                                ),
                                child: Text(
                                  isDone
                                      ? 'Terminé'
                                      : (daysLeft < 0
                                          ? 'En retard'
                                          : '$daysLeft j'),
                                  style: TextStyle(
                                    color: isDone
                                        ? const Color(0xFF22C55E)
                                        : (daysLeft < 0
                                            ? const Color(0xFFE43B3B)
                                            : const Color(0xFFE9B400)),
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
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

class _FilterChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _FilterChip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
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
      ),
    );
  }
}
