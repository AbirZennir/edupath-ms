package com.example.services;

import com.example.entities.Etudiant;
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

    public Utilisateur register(Utilisateur utilisateur) {
        utilisateur.setPassword(passwordEncoder.encode(utilisateur.getPassword()));
        return userRepository.save(utilisateur);
    }

    public Utilisateur findByEmail(String email) {
        Utilisateur utilisateur = userRepository.findByEmail(email);
        if (utilisateur == null) {
            throw new RuntimeException("Aucun étudiant trouvé avec cet email");
        }
        return utilisateur;    }
}
