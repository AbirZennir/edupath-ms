import 'dart:convert';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// Base API configurable via --dart-define=API_BASE_URL=...
// Fallbacks:
//  - Web: http://localhost:8082
//  - Android/iOS simulators: http://10.0.2.2:8082 (Android), http://localhost:8082 otherwise
String _resolveBaseUrl() {
  const envBase = String.fromEnvironment('API_BASE_URL');
  if (envBase.isNotEmpty) return envBase;
  if (kIsWeb) return 'http://localhost:8082';
  if (Platform.isAndroid) return 'http://10.0.2.2:8082';
  return 'http://localhost:8082';
}

final String _apiBaseUrl = _resolveBaseUrl();

class ApiClient {
  ApiClient({http.Client? httpClient}) : _client = httpClient ?? http.Client();

  final http.Client _client;

  Uri _uri(String path) => Uri.parse('$_apiBaseUrl$path');

  Future<String> login({
    required String email,
    required String password,
  }) async {
    final resp = await _client.post(
      _uri('/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );

    if (resp.statusCode != 200) {
      throw Exception('Login failed (${resp.statusCode})');
    }

    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    final token = data['token'] as String?;
    if (token == null || token.isEmpty) {
      throw Exception('Token missing in response');
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('auth_token', token);
    return token;
  }

  Future<void> register({
    required String firstName,
    required String lastName,
    required String email,
    required String password,
  }) async {
    final resp = await _client.post(
      _uri('/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'prenom': firstName,
        'nom': lastName,
        'email': email,
        'password': password,
        'role': 'ETUDIANT',
      }),
    );

    if (resp.statusCode != 200 && resp.statusCode != 201) {
      throw Exception('Registration failed (${resp.statusCode})');
    }
  }

  Future<String?> getSavedToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('auth_token');
  }

  Future<void> clearToken() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('auth_token');
  }
}
