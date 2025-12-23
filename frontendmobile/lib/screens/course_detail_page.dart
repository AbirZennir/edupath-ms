import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:confetti/confetti.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:math';

class CourseDetailPage extends StatefulWidget {
  final String courseTitle;

  const CourseDetailPage({super.key, required this.courseTitle});

  @override
  State<CourseDetailPage> createState() => _CourseDetailPageState();
}

class _CourseDetailPageState extends State<CourseDetailPage> {
  late ConfettiController _confettiController;
  bool _isCompleted = false;
  Map<int, bool> _chapterDoneStatus = {};

  // Base Chapters Definition
  final List<String> _chapterTitles = [
    '1. Introduction et Bases',
    '2. Concepts Avancés',
    '3. Études de Cas Pratiques',
    '4. Projet Final'
  ];

  @override
  void initState() {
    super.initState();
    _confettiController = ConfettiController(duration: const Duration(seconds: 3));
    _loadStatus();
  }

  @override
  void dispose() {
    _confettiController.dispose();
    super.dispose();
  }

  Future<void> _loadStatus() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _isCompleted = prefs.getBool('course_completed_${widget.courseTitle}') ?? false;
      
      // Load chapter statuses
      for (int i = 0; i < _chapterTitles.length; i++) {
        // Default first two chapters are always true (as per original mock), others depend on save
        if (i < 2) {
          _chapterDoneStatus[i] = true; 
        } else {
          _chapterDoneStatus[i] = prefs.getBool('chapter_done_${widget.courseTitle}_$i') ?? false;
        }
      }
    });
  }

  Future<void> _toggleCompletion() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _isCompleted = !_isCompleted;
    });
    await prefs.setBool('course_completed_${widget.courseTitle}', _isCompleted);
    
    if (_isCompleted) {
      _confettiController.play();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Félicitations ! Cours marqué comme terminé 🎓'),
            backgroundColor: Color(0xFF22C55E),
            duration: Duration(seconds: 2),
          ),
        );
        
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) {
             Navigator.of(context).popUntil((route) => route.isFirst);
          }
        });
      }
    }
  }

  String? _getChapterUrl(String chapterTitle) {
    if (chapterTitle.contains('Études de Cas')) {
       switch (widget.courseTitle) {
         case 'Algorithmique': return 'https://www.javascriptdezero.com/blog/apprendre-algorithmique-cas-etude-algorithme/';
         case 'Physique Quantique': return 'https://fad.umi.ac.ma/pluginfile.php/1829/mod_resource/content/16/co/etudecas.html';
         case 'Mathématiques Avancées': return 'https://www.mystudies.com/fr-fr/matieres-scientifiques-et-technologiques/mathematiques/etude-de-cas/';
         case 'Chimie Organique': return 'https://studylibfr.com/doc/2836743/chimie-organique-%E2%80%93-examen';
         case 'Littérature Française': return 'https://www.mystudies.com/fr-ad/philosophie-et-litterature/litterature/etude-de-cas/';
       }
    } else if (chapterTitle.contains('Projet Final')) {
      switch (widget.courseTitle) {
         case 'Algorithmique': return 'https://www-sop.inria.fr/members/Nicolas.Nisse/projetsV2.pdf';
         case 'Physique Quantique': return 'https://prezi.com/p/kluegbvgyvdh/physiques12-projet-finale-alara/';
         case 'Mathématiques Avancées': return 'https://www.ulaval.ca/etudes/cours/mat-3600-projet-de-fin-detudes';
         case 'Chimie Organique': return 'https://www.scribd.com/document/775854180/Chimie-Organique-Finale-1';
         case 'Littérature Française': return 'https://langueetlitteraturefrancaises.com/pfe-et-theses/';
      }
    }
    return null;
  }

  Future<void> _pickAndUploadFile(int index) async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
      );

      if (result != null) {
        // Show loading
        if (!mounted) return;
        showDialog(
          context: context,
          barrierDismissible: false,
          builder: (ctx) => const Center(
            child: CircularProgressIndicator(color: Colors.white),
          ),
        );

        // Simulate upload delay
        await Future.delayed(const Duration(seconds: 2));

        // Mark as done
        final prefs = await SharedPreferences.getInstance();
        setState(() {
          _chapterDoneStatus[index] = true;
        });
        await prefs.setBool('chapter_done_${widget.courseTitle}_$index', true);

        if (!mounted) return;
        Navigator.pop(context); // Close loading dialog
        Navigator.pop(context); // Close bottom sheet if open (optional, maybe keep it open to show updated status)
        
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Devoir envoyé avec succès !'),
            backgroundColor: Color(0xFF22C55E),
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erreur: $e')),
      );
    }
  }

  void _showSubmissionOptions(String title, int index, String? url) {
    final isDone = _chapterDoneStatus[index] ?? false;

    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Container(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      title,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: isDone ? const Color(0xFFDFF9EB) : const Color(0xFFFDE2E1),
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      isDone ? 'Rendu ✅' : 'Non rendu',
                      style: TextStyle(
                        color: isDone ? const Color(0xFF22C55E) : const Color(0xFFE43B3B),
                        fontWeight: FontWeight.w700,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              if (url != null)
                ListTile(
                  leading: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: const Color(0xFFEFF6FF),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.description_outlined, color: Color(0xFF2F65D9)),
                  ),
                  title: const Text('Consulter le sujet'),
                  subtitle: const Text('Ouvrir le document PDF ou Lien'),
                  onTap: () async {
                    Navigator.pop(context);
                    final uri = Uri.parse(url);
                    if (await canLaunchUrl(uri)) {
                      await launchUrl(uri, mode: LaunchMode.externalApplication);
                    }
                  },
                ),
              const SizedBox(height: 12),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF1F2F6),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(Icons.upload_file, color: Color(0xFF4B5563)),
                ),
                title: const Text('Déposer mon travail'),
                subtitle: const Text('Format PDF uniquement'),
                onTap: () {
                   // Keep bottom sheet open or close? Maybe close it first to avoid context issues with dialogs
                   // Or call directly.
                   _pickAndUploadFile(index);
                },
              ),
              const SizedBox(height: 24),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    final assignments = [
      ('Devoir Chapitre 3', '2 jours', const Color(0xFFFDE2E1),
          const Color(0xFFE43B3B)),
      ('Exercices Matrices', '1 semaine', const Color(0xFFE8F0FF),
          const Color(0xFF2F65D9)),
    ];

    return Scaffold(
      backgroundColor: theme.scaffoldBackgroundColor,
      body: SafeArea(
        child: Stack(
          alignment: Alignment.topCenter,
          children: [
            Column(
              children: [
                _header(context),
                Expanded(
                  child: ListView(
                    padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                    children: [
                      _sectionTitle(context, 'Description du cours'),
                      _card(
                        child: Text(
                          'Ce cours couvre les concepts avancés en mathématiques incluant l’algèbre linéaire, le calcul différentiel et intégral, ainsi que les équations différentielles.',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: theme.brightness == Brightness.dark ? Colors.grey[300] : const Color(0xFF4B5563),
                            height: 1.4,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      _sectionTitle(context, 'Chapitres'),
                      _card(
                        child: Column(
                          children: [
                            ..._chapterTitles.asMap().entries.map(
                              (entry) {
                                final index = entry.key;
                                final title = entry.value;
                                final isDone = _chapterDoneStatus[index] ?? false;
                                return _chapterTile(
                                  index: index + 1,
                                  title: title,
                                  done: isDone,
                                  isLast: index == _chapterTitles.length - 1,
                                );
                              },
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      _sectionTitle(context, 'Devoirs du cours'),
                      _card(
                        child: Column(
                          children: assignments
                              .map(
                                (a) => _assignmentTile(
                                  title: a.$1,
                                  deadline: a.$2,
                                  badgeColor: a.$3,
                                  badgeTextColor: a.$4,
                                ),
                              )
                              .toList(),
                        ),
                      ),
                      const SizedBox(height: 24),
                      Row(
                        children: [
                          Expanded(
                            child: SizedBox(
                              height: 48,
                              child: OutlinedButton.icon(
                                onPressed: _toggleCompletion,
                                style: OutlinedButton.styleFrom(
                                  backgroundColor: _isCompleted ? const Color(0xFFDFF9EB) : Colors.transparent,
                                  side: BorderSide(
                                    color: _isCompleted ? const Color(0xFF22C55E) : theme.colorScheme.primary,
                                  ),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                ),
                                icon: Icon(
                                  _isCompleted ? Icons.check_circle : Icons.check_circle_outline, 
                                  color: _isCompleted ? const Color(0xFF22C55E) : theme.colorScheme.primary
                                ),
                                label: Text(
                                  _isCompleted ? 'Terminé' : 'Marquer terminé',
                                  style: TextStyle(
                                    fontWeight: FontWeight.w700,
                                    color: _isCompleted ? const Color(0xFF22C55E) : theme.colorScheme.primary,
                                  ),
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: SizedBox(
                              height: 48,
                              child: ElevatedButton.icon(
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: const Color(0xFF2F65D9),
                                  foregroundColor: Colors.white,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                ),
                                onPressed: () async {
                                  // Map course titles to real Youtube videos
                                  String url;
                                  switch (widget.courseTitle) {
                                    case 'Physique Quantique':
                                      url = 'https://www.youtube.com/watch?v=3-Ps97M_BaY&list=PLXTROeoOYnlwkScp2q3_AS_l9APnRxUdc';
                                      break;
                                    case 'Algorithmique':
                                      url = 'https://www.youtube.com/watch?v=m_yj2p6SCEM&list=PLZpzLuUp9qXwrApSukhtvpi4U6l-INcTI';
                                      break;
                                    case 'Chimie Organique':
                                      url = 'https://www.youtube.com/watch?v=ZCFEqsxlfQc&list=PLUPfbbGYbRqDgoftc9nDb_Dc8J6pEl2R9';
                                      break;
                                     case 'Mathématiques Avancées':
                                      url = 'https://www.youtube.com/watch?v=UubU3U2C8WM&list=PLybg94GvOJ9GkwyYc3jolcsDgjl2Niarg';
                                      break;
                                    case 'Littérature Française':
                                      url = 'https://www.youtube.com/watch?v=mAaohxfxXKM&list=PLHFU8yWQ83n6hRmegpA8YuAGUzjIpU61f';
                                      break;
                                    default:
                                      url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'; // Default placeholder
                                  }

                                  final uri = Uri.parse(url);
                                  // Launch with external application mode to try opening Youtube app directly
                                  if (await canLaunchUrl(uri)) {
                                    await launchUrl(uri, mode: LaunchMode.externalApplication);
                                  } else {
                                    if (context.mounted) {
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        const SnackBar(content: Text('Impossible de lancer la vidéo')),
                                      );
                                    }
                                  }
                                },
                                icon: const Icon(Icons.play_arrow),
                                label: const Text(
                                  'Commencer',
                                  style: TextStyle(fontWeight: FontWeight.w700),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
            ConfettiWidget(
              confettiController: _confettiController,
              blastDirectionality: BlastDirectionality.explosive,
              shouldLoop: false,
              colors: const [Colors.green, Colors.blue, Colors.pink, Colors.orange, Colors.purple],
            ),
          ],
        ),
      ),
    );
  }

  Widget _header(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 12, 8, 18),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFFF16068), Color(0xFFF2777F)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.vertical(
          bottom: Radius.circular(24),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context, true), // Return true to signal potential refresh
            icon: const Icon(Icons.arrow_back, color: Colors.white),
          ),
          const SizedBox(width: 8),
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(14),
            ),
            child:
                const Icon(Icons.straighten, color: Colors.white, size: 26),
          ),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                widget.courseTitle,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 4),
              const Text(
                'Prof. Martin Dubois',
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 13,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(BuildContext context, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        text,
        style: Theme.of(context).textTheme.titleMedium?.copyWith(
          fontWeight: FontWeight.w700,
          fontSize: 15,
        ),
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: child,
      ),
    );
  }

  Widget _chapterTile({
    required int index,
    required String title,
    required bool done,
    required bool isLast,
  }) {
    final url = _getChapterUrl(title);
    
    return Column(
      children: [
        InkWell(
          onTap: () {
            if (url != null) {
              _showSubmissionOptions(title, index - 1, url);
            }
          },
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
            child: Row(
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: done
                        ? const Color(0xFFDFF9EB)
                        : const Color(0xFFF1F2F6),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  alignment: Alignment.center,
                  child: done 
                    ? const Icon(Icons.check, size: 16, color: Color(0xFF22C55E))
                    : Text(
                        '$index',
                        style: const TextStyle(
                          color: Color(0xFF6B7280),
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                      color: done ? Colors.black87 : (url != null ? const Color(0xFF2F65D9) : Colors.black54),
                      decoration: done ? TextDecoration.lineThrough : null,
                    ),
                  ),
                ),
                if (done)
                  const Icon(Icons.check_circle, color: Color(0xFF22C55E), size: 20)
                else if (url != null)
                   const Icon(Icons.upload_file, color: Color(0xFF2F65D9), size: 20)
                else
                   const Icon(Icons.lock_outline, color: Color(0xFF9CA3AF), size: 20),
              ],
            ),
          ),
        ),
        if (!isLast)
          const Divider(
            height: 16,
            thickness: 1,
            color: Color(0xFFE5E7EB),
          ),
      ],
    );
  }

  Widget _assignmentTile({
    required String title,
    required String deadline,
    required Color badgeColor,
    required Color badgeTextColor,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                children: const [
                  Icon(Icons.access_time,
                      size: 14, color: Color(0xFF6B7280)),
                  SizedBox(width: 6),
                  Text(
                    'À rendre bientôt',
                    style: TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ],
          ),
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: badgeColor,
              borderRadius: BorderRadius.circular(999),
            ),
            child: Text(
              deadline,
              style: TextStyle(
                color: badgeTextColor,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
