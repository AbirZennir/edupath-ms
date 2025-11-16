import 'package:flutter/material.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final _formKey = GlobalKey<FormState>();

  String? _errorMessage;

  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _firstNameController = TextEditingController();
  final _lastNameController = TextEditingController();

  bool _acceptedTerms = false;

  void _register() {
    if (_formKey.currentState!.validate()) {
      if (_acceptedTerms) {
        setState(() {
          _errorMessage = null;
        });
        // Submit registration
      } else {
        setState(() {
          _errorMessage = "Veuillez accepter les conditions.";
        });
      }
    } else {
      setState(() {
        _errorMessage = "Veuillez vérifier les informations saisies.";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Card(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Form(
                key: _formKey,
                child: Column(
                  children: [
                    const Icon(Icons.school, size: 50, color: Colors.blue),
                    const SizedBox(height: 10),
                    const Text(
                      "Créer un compte StudentCoach",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 5),
                    const Text(
                      "Rejoignez votre espace d'apprentissage personnalisé",
                    ),
                    const SizedBox(height: 20),
                    TextFormField(
                      controller: _firstNameController,
                      decoration: const InputDecoration(
                        labelText: 'Prénom',
                        prefixIcon: Icon(Icons.person),
                      ),
                      validator: (value) =>
                          value!.isEmpty ? "Champ requis" : null,
                    ),
                    const SizedBox(height: 10),
                    TextFormField(
                      controller: _lastNameController,
                      decoration: const InputDecoration(
                        labelText: 'Nom',
                        prefixIcon: Icon(Icons.person_2),
                      ),
                      validator: (value) =>
                          value!.isEmpty ? "Champ requis" : null,
                    ),
                    const SizedBox(height: 10),
                    TextFormField(
                      controller: _emailController,
                      decoration: const InputDecoration(
                        labelText: 'Adresse e-mail',
                        prefixIcon: Icon(Icons.email),
                      ),
                      validator: (value) =>
                          value!.isEmpty ? "Champ requis" : null,
                    ),
                    const SizedBox(height: 10),
                    TextFormField(
                      controller: _passwordController,
                      obscureText: true,
                      decoration: const InputDecoration(
                        labelText: 'mot de passe',
                        prefixIcon: Icon(Icons.lock),
                      ),
                      validator: (value) =>
                          value!.length < 6 ? "Min 6 caractères" : null,
                    ),
                    const SizedBox(height: 10),
                    TextFormField(
                      controller: _confirmPasswordController,
                      obscureText: true,
                      decoration: const InputDecoration(
                        labelText: 'Confirmation de mot de passe',
                        prefixIcon: Icon(Icons.lock),
                      ),
                      validator: (value) => value != _passwordController.text
                          ? "Les mots de passe ne correspondent pas"
                          : null,
                    ),
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        Checkbox(
                          value: _acceptedTerms,
                          onChanged: (value) {
                            setState(() {
                              _acceptedTerms = value!;
                            });
                          },
                        ),
                        const Expanded(
                          child: Text(
                            "J'accepte les Conditions d'utilisation et la Politique de confidentialité",
                          ),
                        ),
                      ],
                    ),
                    if (_errorMessage != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8.0),
                        child: Text(
                          _errorMessage!,
                          style: const TextStyle(color: Colors.red),
                        ),
                      ),
                    const SizedBox(height: 20),
                    ElevatedButton.icon(
                      onPressed: _register,
                      icon: const Icon(Icons.app_registration),
                      label: const Text("Créer mon compte"),
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
                      child: const Text("Déjà un compte ? Se connecter"),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
