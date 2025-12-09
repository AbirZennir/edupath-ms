package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AtRiskStudentDto {
    private int id;
    private String nom;
    private String avatar;          // initiales
    private String classe;          // ex: L3 Informatique
    private List<String> modules;   // ex: ["Algorithmes", "Base de données"]
    private String niveau;          // elevé | moyen
    private String derniereConnexion; // ex: "2 jours"
    private int risque;             // probabilité d'échec en %
}
