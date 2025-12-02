import 'package:flutter/material.dart';
import 'reset_password_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  bool _invalidCredentials = false;

  void _login() {
    setState(() {
      _invalidCredentials = true; // Simuler une erreur
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Card(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                children: [
                  const Icon(Icons.school, size: 50, color: Colors.blue),
                  const SizedBox(height: 10),
                  const Text(
                    "Connexion à StudentCoach",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 5),
                  const Text(
                    "Accédez à votre espace d'apprentissage personnalisé",
                  ),
                  const SizedBox(height: 20),
                  TextField(
                    controller: _emailController,
                    decoration: const InputDecoration(
                      labelText: 'Adresse e-mail',
                      prefixIcon: Icon(Icons.email),
                    ),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: _passwordController,
                    obscureText: true,
                    decoration: const InputDecoration(
                      labelText: 'Mot de passe',
                      prefixIcon: Icon(Icons.lock),
                    ),
                  ),
                  if (_invalidCredentials)
                    const Padding(
                      padding: EdgeInsets.only(top: 10),
                      child: Text(
                        "Identifiants invalides. Veuillez vérifier vos informations.",
                        style: TextStyle(color: Colors.red),
                      ),
                    ),
                  const SizedBox(height: 20),
                  Row(
                    children: [
                      Checkbox(value: false, onChanged: (val) {}),
                      const Text("Se souvenir de moi"),
                    ],
                  ),
                  const SizedBox(height: 10),
                  ElevatedButton.icon(
                    onPressed: _login,
                    icon: const Icon(Icons.login),
                    label: const Text("Se connecter"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  TextButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    child: const Text("Pas encore de compte ? Créez-en un"),
                  ),
                  const SizedBox(height: 6),
                  TextButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const ResetPasswordPage(),
                        ),
                      );
                    },
                    child: const Text("Mot de passe oublié ?"),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
