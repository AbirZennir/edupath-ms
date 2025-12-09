import 'package:flutter/material.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        title: const Text('Profil'),
        backgroundColor: const Color(0xFFF5F7FB),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Center(
            child: Column(
              children: const [
                CircleAvatar(
                  radius: 40,
                  child: Icon(Icons.person, size: 40),
                ),
                SizedBox(height: 8),
                Text(
                  'Sophie Martin',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 18,
                  ),
                ),
                SizedBox(height: 4),
                Text('sophie.martin@example.com'),
              ],
            ),
          ),
          const SizedBox(height: 24),
          const ListTile(
            leading: Icon(Icons.person_outline),
            title: Text('Informations du profil'),
          ),
          const Divider(height: 0),
          const ListTile(
            leading: Icon(Icons.lock_outline),
            title: Text('Changer le mot de passe'),
          ),
          const Divider(height: 0),
          const ListTile(
            leading: Icon(Icons.tune),
            title: Text('Préférences'),
          ),
          const Divider(height: 0),
          const SizedBox(height: 24),
          ListTile(
            leading: const Icon(Icons.logout, color: Colors.red),
            title: const Text(
              'Déconnexion',
              style: TextStyle(color: Colors.red),
            ),
            onTap: () {
              Navigator.pushReplacementNamed(context, '/login');
            },
          ),
        ],
      ),
    );
  }
}
