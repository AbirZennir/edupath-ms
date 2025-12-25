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
    private String role;
}
