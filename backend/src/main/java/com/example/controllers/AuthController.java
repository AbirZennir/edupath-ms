package com.example.controllers;

import com.example.entities.Utilisateur;
import com.example.repositories.UserRepository;
import com.example.security.JwtUtils;
import com.example.services.UtilisateurService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private UtilisateurService utilisateurService;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    JwtUtils jwtUtils;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @PostMapping("/register")
    public Utilisateur register(@RequestBody Utilisateur utilisateur){
        return utilisateurService.register(utilisateur);
    }

    @PostMapping("/login")
    public Map<String,String> login(@RequestBody Map<String,String> loginData){
        String email = loginData.get("email");
        String password = loginData.get("password");
        Utilisateur utilisateur = utilisateurService.findByEmail(email);
        if (utilisateur == null || !passwordEncoder.matches(password, utilisateur.getPassword())) {
            throw new RuntimeException("Email ou mot de passe incorrect");
        }
        String token = jwtUtils.generateToken(email);
        return Map.of("token", token);
    }
}
