package com.example.services;

import com.example.dto.AtRiskStudentDto;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

@Service
public class AtRiskStudentsService {

    private final List<AtRiskStudentDto> seed = List.of(
            new AtRiskStudentDto(1, "Dubois Alexandre", "AD", "L3 Informatique", List.of("Algorithmes", "Base de données"), "elevé", "2 jours", 78),
            new AtRiskStudentDto(2, "Martin Sophie", "MS", "L2 Physique", List.of("Mécanique"), "moyen", "1 jour", 65),
            new AtRiskStudentDto(3, "Bernard Lucas", "BL", "L3 Informatique", List.of("Algorithmes"), "elevé", "3 jours", 72),
            new AtRiskStudentDto(4, "Petit Emma", "PE", "M1 Data Science", List.of("Machine Learning"), "moyen", "5 heures", 58),
            new AtRiskStudentDto(5, "Roux Thomas", "RT", "L3 Informatique", List.of("Algorithmes", "Réseaux"), "elevé", "4 jours", 81),
            new AtRiskStudentDto(6, "Moreau Léa", "ML", "L2 Mathématiques", List.of("Algèbre linéaire"), "moyen", "1 jour", 62),
            new AtRiskStudentDto(7, "Simon Hugo", "SH", "L3 Informatique", List.of("Base de données"), "moyen", "2 jours", 69),
            new AtRiskStudentDto(8, "Laurent Chloé", "LC", "M2 IA", List.of("Deep Learning"), "elevé", "3 jours", 75),
            new AtRiskStudentDto(9, "Fournier Louis", "FL", "L3 Informatique", List.of("Algorithmes"), "moyen", "1 jour", 67),
            new AtRiskStudentDto(10, "Michel Sarah", "MS", "L2 Physique", List.of("Mécanique", "Thermodynamique"), "elevé", "5 jours", 73)
    );

    public List<AtRiskStudentDto> list(String niveau) {
        return seed.stream()
                .filter(s -> isBlank(niveau) || s.getNiveau().equalsIgnoreCase(niveau))
                .collect(Collectors.toList());
    }

    public long countByLevel(String niveau) {
        return seed.stream().filter(s -> s.getNiveau().equalsIgnoreCase(niveau)).count();
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
