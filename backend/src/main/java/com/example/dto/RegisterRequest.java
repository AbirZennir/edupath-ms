package com.example.dto;

import lombok.Data;

@Data
public class RegisterRequest {
    private String prenom;
    private String nom;
    private String email;
    private String password;
    private String departement;
    private String institution;
    // Accepte des valeurs comme "PROFESSEUR" ou "ENSEIGNANT" (qui seront mappées côté service)
    private String role;
}
