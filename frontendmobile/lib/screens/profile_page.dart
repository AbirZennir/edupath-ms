import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../api_client.dart';
import '../preferences_service.dart';

class ProfilePage extends StatefulWidget {
  final String studentName;
  final String studentEmail;

  const ProfilePage({
    super.key,
    this.studentName = 'Étudiant',
    this.studentEmail = 'etudiant@edupath.fr',
  });

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  final ApiClient _apiClient = ApiClient();
  final PreferencesService _prefs = PreferencesService();
  
  // State variables
  late String _displayName;
  late String _displayEmail; 
  String _major = 'Chargement...';

  @override
  void initState() {
    super.initState();
    _displayName = widget.studentName;
    _displayEmail = widget.studentEmail;
    _loadProfileData();
  }

  // Update display if parent widget updates props
  @override
  void didUpdateWidget(ProfilePage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.studentName != oldWidget.studentName) {
      _displayName = widget.studentName;
    }
  }

  Future<void> _loadProfileData() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _major = prefs.getString('user_major') ?? 'Informatique';
    });
  }

  void _showEditProfileDialog() {
    final nameController = TextEditingController(text: _displayName);
    String tempMajor = _major;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            title: Text(_prefs.language == 'English' ? 'Edit Profile' : 'Modifier le profil'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  decoration: InputDecoration(
                    labelText: _prefs.language == 'English' ? 'Full Name' : 'Nom complet'
                  ),
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<String>(
                  value: ['Informatique', 'Mathématiques', 'Physique'].contains(tempMajor) ? tempMajor : 'Informatique',
                  items: const [
                    DropdownMenuItem(value: 'Informatique', child: Text('Informatique')),
                    DropdownMenuItem(value: 'Mathématiques', child: Text('Mathématiques')),
                    DropdownMenuItem(value: 'Physique', child: Text('Physique')),
                  ],
                  onChanged: (v) {
                    if (v != null) setDialogState(() => tempMajor = v);
                  },
                  decoration: InputDecoration(
                    labelText: _prefs.language == 'English' ? 'Major' : 'Filière'
                  ),
                ),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text(_prefs.language == 'English' ? 'Cancel' : 'Annuler'),
              ),
              FilledButton(
                onPressed: () {
                  setState(() {
                    _displayName = nameController.text;
                    _major = tempMajor;
                  });
                  _prefs.setMajor(tempMajor);
                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text(_prefs.language == 'English' ? 'Profile updated' : 'Profil mis à jour')),
                  );
                },
                child: Text(_prefs.language == 'English' ? 'Save' : 'Enregistrer'),
              ),
            ],
          );
        },
      ),
    );
  }

  void _showChangePasswordDialog() {
    final oldPassController = TextEditingController();
    final newPassController = TextEditingController();
    bool _isLoading = false;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            title: Text(_prefs.language == 'English' ? 'Change Password' : 'Changer mot de passe'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (_isLoading)
                  const SizedBox(
                    height: 50,
                    child: Center(child: CircularProgressIndicator()),
                  )
                else ...[
                  TextField(
                    controller: oldPassController,
                    obscureText: true,
                    decoration: InputDecoration(
                        labelText: _prefs.language == 'English' ? 'Old Password' : 'Ancien mot de passe'
                    ),
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: newPassController,
                    obscureText: true,
                    decoration: InputDecoration(
                        labelText: _prefs.language == 'English' ? 'New Password' : 'Nouveau mot de passe'
                    ),
                  ),
                ]
              ],
            ),
            actions: _isLoading ? [] : [
               TextButton(
                 onPressed: () => Navigator.pop(context),
                 child: Text(_prefs.language == 'English' ? 'Cancel' : 'Annuler'),
               ),
               FilledButton(
                 onPressed: () async {
                   if (newPassController.text.isEmpty || oldPassController.text.isEmpty) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text(_prefs.language == 'English' ? 'Please fill all fields' : 'Veuillez remplir tous les champs')),
                      );
                      return;
                   }
                   
                   setDialogState(() => _isLoading = true);
                   
                   // Simulate network request
                   await Future.delayed(const Duration(seconds: 2));
                   
                   if (!mounted) return;
                   Navigator.pop(context);
                   ScaffoldMessenger.of(context).showSnackBar(
                     SnackBar(content: Text(_prefs.language == 'English' ? 'Password changed successfully' : 'Mot de passe changé avec succès')),
                   );
                 },
                 child: Text(_prefs.language == 'English' ? 'Confirm' : 'Confirmer'),
               )
            ],
          );
        },
      ),
    );
  }

  void _logout() async {
    await _apiClient.clearToken();
    if (!mounted) return;
    Navigator.of(context, rootNavigator: true).pushReplacementNamed('/login');
  }

  @override
  Widget build(BuildContext context) {
    // Listen to changes to rebuild translations/theme
    return AnimatedBuilder(
      animation: _prefs,
      builder: (context, _) => Scaffold(
        backgroundColor: Theme.of(context).scaffoldBackgroundColor,
        body: SafeArea(
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header Gradient
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Color(0xFF2F65D9), Color(0xFF5D93FF)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.only(
                      bottomLeft: Radius.circular(32),
                      bottomRight: Radius.circular(32),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _prefs.language == 'English' ? 'My Profile' : 'Mon Profil',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
                
                const SizedBox(height: 20),

                // Profile Card
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Row(
                        children: [
                          CircleAvatar(
                            radius: 30,
                            backgroundColor: const Color(0xFFEFF6FF),
                            child: const Icon(Icons.person_outline,
                                size: 32, color: Color(0xFF2F65D9)),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  _displayName,
                                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                    fontWeight: FontWeight.bold
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'Master 1 - $_major',
                                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                    color: Colors.grey
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _displayEmail,
                                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: Colors.grey
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // Settings Sections
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                          _prefs.language == 'English' ? 'Account' : 'Compte',
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold)
                      ),
                      const SizedBox(height: 8),
                      _SettingItem(
                        icon: Icons.edit_outlined,
                        title: _prefs.language == 'English' ? 'Edit Profile' : 'Modifier le profil',
                        onTap: _showEditProfileDialog,
                      ),
                      _SettingItem(
                        icon: Icons.lock_outline,
                        title: _prefs.language == 'English' ? 'Change Password' : 'Changer le mot de passe',
                        onTap: _showChangePasswordDialog,
                      ),

                      const SizedBox(height: 24),
                       Text(
                          _prefs.language == 'English' ? 'Preferences' : 'Préférences',
                          style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold)
                      ),
                      const SizedBox(height: 8),
                      _SettingSwitch(
                        icon: Icons.notifications_outlined,
                        title: 'Notifications',
                        value: _prefs.notificationsEnabled,
                        onChanged: (v) {
                          _prefs.setNotifications(v);
                        },
                      ),
                      _SettingDropdown(
                        icon: Icons.language,
                        title: _prefs.language == 'English' ? 'Language' : 'Langue',
                        value: _prefs.language,
                        items: const ['Français', 'English'], // Spanish Removed
                        onChanged: (v) {
                           if (v != null) {
                             _prefs.setLanguage(v);
                           }
                        },
                      ),
                      _SettingDropdown(
                        icon: Icons.palette_outlined,
                        title: _prefs.language == 'English' ? 'Theme' : 'Thème',
                        value: _prefs.themeMode,
                        items: const ['Clair', 'Sombre'], // Removed system to simplify
                        onChanged: (v) {
                           if (v != null) {
                             _prefs.setThemeMode(v);
                           }
                        },
                      ),

                      const SizedBox(height: 32),
                      SizedBox(
                        width: double.infinity,
                        height: 50,
                        child: ElevatedButton(
                          onPressed: _logout,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFEF4444),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 0,
                          ),
                          child: Text(
                            _prefs.language == 'English' ? 'Logout' : 'Se déconnecter',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ... _SettingItem and others ...
class _SettingItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final VoidCallback onTap;

  const _SettingItem({
    required this.icon,
    required this.title,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      // Use Card or ensure Container respects theme
      child: Card(
        margin: EdgeInsets.zero,
        child: ListTile(
          leading: Icon(icon),
          title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
          trailing: const Icon(Icons.chevron_right),
          onTap: onTap,
        ),
      ),
    );
  }
}

class _SettingSwitch extends StatelessWidget {
  final IconData icon;
  final String title;
  final bool value;
  final ValueChanged<bool> onChanged;

  const _SettingSwitch({
    required this.icon,
    required this.title,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      child: Card(
        margin: EdgeInsets.zero,
        child: ListTile(
          leading: Icon(icon),
          title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
          trailing: Switch(
            value: value,
            onChanged: onChanged,
            activeColor: Theme.of(context).primaryColor,
          ),
        ),
      ),
    );
  }
}

class _SettingDropdown extends StatelessWidget {
  final IconData icon;
  final String title;
  final String value;
  final List<String> items;
  final ValueChanged<String?> onChanged;

  const _SettingDropdown({
    required this.icon,
    required this.title,
    required this.value,
    required this.items,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      child: Card(
        margin: EdgeInsets.zero,
        child: ListTile(
          leading: Icon(icon),
          title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
          trailing: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: value,
              items: items.map((String item) {
                return DropdownMenuItem<String>(
                  value: item,
                  child: Text(item, style: const TextStyle(fontSize: 14)),
                );
              }).toList(),
              onChanged: onChanged,
            ),
          ),
        ),
      ),
    );
  }
}
