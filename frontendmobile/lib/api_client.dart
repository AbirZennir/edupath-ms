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
    final userId = data['userId'] as int?; // Added userId

    if (token == null || token.isEmpty) {
      throw Exception('Token missing in response');
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('auth_token', token);
    if (userId != null) {
      await prefs.setInt('auth_user_id', userId);
    }
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

    // Auto-login after registration if token is provided
    try {
      final data = jsonDecode(resp.body) as Map<String, dynamic>;
      final token = data['token'] as String?;
      final userId = data['userId'] as int?;

      if (token != null && token.isNotEmpty) {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('auth_token', token);
        if (userId != null) {
          await prefs.setInt('auth_user_id', userId); // Persist userId
        }
      }
    } catch (_) {
      // Ignore parsing errors for register response
    }
  }

  Future<String?> getSavedToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('auth_token');
  }

  Future<int?> getSavedUserId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt('auth_user_id');
  }

  Future<Map<String, dynamic>> getDashboard(int studentId) async {
    final resp = await _client.get(
      _uri('/dashboard/$studentId'),
      headers: {'Content-Type': 'application/json'},
    );

    if (resp.statusCode != 200) {
      throw Exception('Failed to load dashboard (${resp.statusCode})');
    }

    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<List<Map<String, dynamic>>> getCourses(int studentId) async {
    final resp = await _client.get(
      _uri('/courses/student/$studentId'),
      headers: {'Content-Type': 'application/json'},
    );

    if (resp.statusCode != 200) {
      throw Exception('Failed to load courses (${resp.statusCode})');
    }

    final List<dynamic> data = jsonDecode(resp.body);
    return data.cast<Map<String, dynamic>>();
  }

  Future<List<Map<String, dynamic>>> getAssignments(int studentId) async {
    final resp = await _client.get(
      _uri('/assignments/student/$studentId'),
      headers: {'Content-Type': 'application/json'},
    );

    if (resp.statusCode != 200) {
      throw Exception('Failed to load assignments (${resp.statusCode})');
    }

    final List<dynamic> data = jsonDecode(resp.body);
    return data.cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>> getGrades(int studentId) async {
    final resp = await _client.get(
      _uri('/grades/$studentId'),
      headers: {'Content-Type': 'application/json'},
    );

    if (resp.statusCode != 200) {
      throw Exception('Failed to load grades (${resp.statusCode})');
    }

    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<void> clearToken() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('auth_token');
  }
}
