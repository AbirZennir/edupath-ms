package com.example.services;

import com.example.dto.RegisterRequest;
import com.example.entities.Role;
import com.example.entities.Utilisateur;
import com.example.repositories.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class UtilisateurService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    public Utilisateur register(RegisterRequest request) {
        if (userRepository.findByEmail(request.getEmail()) != null) {
            throw new RuntimeException("Email déjà utilisé");
        }
        Utilisateur utilisateur = new Utilisateur();
        utilisateur.setPrenom(request.getPrenom());
        utilisateur.setNom(request.getNom());
        utilisateur.setEmail(request.getEmail());
        utilisateur.setPassword(passwordEncoder.encode(request.getPassword()));
        utilisateur.setAdresse(request.getInstitution());
        utilisateur.setTelephone(null);
        utilisateur.setGenre(null);
        utilisateur.setRole(mapRole(request.getRole()));
        utilisateur.setDateNaissance(null);
        return userRepository.save(utilisateur);
    }

    public Utilisateur findByEmail(String email) {
        Utilisateur utilisateur = userRepository.findByEmail(email);
        if (utilisateur == null) {
            throw new RuntimeException("Aucun utilisateur trouvé avec cet email");
        }
        return utilisateur;
    }

    private Role mapRole(String rawRole) {
        if (rawRole == null || rawRole.isBlank()) {
            return Role.PROFESSEUR;
        }
        String normalized = rawRole.trim().toUpperCase();
        if (normalized.equals("ENSEIGNANT")) {
            return Role.PROFESSEUR;
        }
        try {
            return Role.valueOf(normalized);
        } catch (IllegalArgumentException ex) {
            return Role.PROFESSEUR;
        }
    }
}
