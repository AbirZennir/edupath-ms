import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class PreferencesService extends ChangeNotifier {
  static final PreferencesService _instance = PreferencesService._internal();
  factory PreferencesService() => _instance;
  PreferencesService._internal();

  String _language = 'Français';
  String _themeMode = 'Clair';
  bool _notificationsEnabled = true;
  String _major = 'Informatique';

  String get language => _language;
  String get themeMode => _themeMode;
  bool get notificationsEnabled => _notificationsEnabled;
  String get major => _major;

  bool get isDarkMode => _themeMode == 'Sombre';

  Future<void> loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    _language = prefs.getString('app_language') ?? 'Français';
    _themeMode = prefs.getString('app_theme') ?? 'Clair';
    _notificationsEnabled = prefs.getBool('notifications_enabled') ?? true;
    _major = prefs.getString('user_major') ?? 'Informatique';
    notifyListeners();
  }

  Future<void> setLanguage(String lang) async {
    _language = lang;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('app_language', lang);
    notifyListeners();
  }

  Future<void> setThemeMode(String mode) async {
    _themeMode = mode;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('app_theme', mode);
    notifyListeners();
  }
  
  Future<void> setNotifications(bool enabled) async {
    _notificationsEnabled = enabled;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('notifications_enabled', enabled);
    notifyListeners();
  }

  Future<void> setMajor(String newMajor) async {
    _major = newMajor;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user_major', newMajor);
    notifyListeners();
  }
}
