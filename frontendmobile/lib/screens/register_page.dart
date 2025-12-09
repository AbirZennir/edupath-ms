import 'package:flutter/material.dart';

import '../api_client.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final _formKey = GlobalKey<FormState>();

  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _nameController = TextEditingController();
  final ApiClient _apiClient = ApiClient();

  bool _loading = false;
  String? _errorMessage;

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _loading = true;
      _errorMessage = null;
    });
    try {
      await _apiClient.register(
        firstName: _nameController.text.trim(),
        lastName: '',
        email: _emailController.text.trim(),
        password: _passwordController.text,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Compte créé, vous pouvez vous connecter')),
      );
      Navigator.pushReplacementNamed(context, '/login');
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Inscription échouée: ${e.toString()}';
      });
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Form(
            key: _formKey,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Créer un compte étudiant',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Rejoignez EduPath dès aujourd\'hui',
                  style: theme.textTheme.bodyMedium,
                ),
                const SizedBox(height: 32),
                TextFormField(
                  controller: _nameController,
                  decoration: const InputDecoration(
                    hintText: 'Nom complet',
                    prefixIcon: Icon(Icons.person_outline),
                  ),
                  validator: (v) =>
                      (v == null || v.isEmpty) ? 'Nom requis' : null,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _emailController,
                  decoration: const InputDecoration(
                    hintText: 'Email',
                    prefixIcon: Icon(Icons.mail_outlined),
                  ),
                  validator: (v) =>
                      (v == null || v.isEmpty) ? 'Email requis' : null,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _passwordController,
                  obscureText: true,
                  decoration: const InputDecoration(
                    hintText: 'Mot de passe',
                    prefixIcon: Icon(Icons.lock_outline),
                  ),
                  validator: (v) =>
                      (v == null || v.length < 6) ? 'Min 6 caractères' : null,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  controller: _confirmPasswordController,
                  obscureText: true,
                  decoration: const InputDecoration(
                    hintText: 'Confirmer mot de passe',
                    prefixIcon: Icon(Icons.lock_outline),
                  ),
                  validator: (v) => v != _passwordController.text
                      ? 'Les mots de passe ne correspondent pas'
                      : null,
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<String>(
                  items: const [
                    DropdownMenuItem(
                      value: 'Informatique',
                      child: Text('Informatique'),
                    ),
                    DropdownMenuItem(
                      value: 'Mathématiques',
                      child: Text('Mathématiques'),
                    ),
                    DropdownMenuItem(
                      value: 'Physique',
                      child: Text('Physique'),
                    ),
                  ],
                  onChanged: (_) {},
                  decoration: const InputDecoration(
                    hintText: 'Classe ou filière',
                  ),
                ),
                if (_errorMessage != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: Text(
                      _errorMessage!,
                      style: const TextStyle(color: Colors.red),
                    ),
                  ),
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  height: 48,
                  child: FilledButton(
                    onPressed: _loading ? null : _submit,
                    child: _loading
                        ? const SizedBox(
                            height: 18,
                            width: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Text('Créer mon compte'),
                  ),
                ),
                const SizedBox(height: 20),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text('Déjà un compte ? '),
                    GestureDetector(
                      onTap: _loading
                          ? null
                          : () => Navigator.pushReplacementNamed(
                                context,
                                '/login',
                              ),
                      child: const Text(
                        'Se connecter',
                        style: TextStyle(
                          color: Color(0xFF2563EB),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
