package com.example.services;

import com.example.dto.AssignmentItemDto;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

@Service
public class AssignmentsService {

    private final List<AssignmentItemDto> seed = List.of(
            new AssignmentItemDto("Devoir Mathématiques Ch.5", "Mathématiques", 2, "pending"),
            new AssignmentItemDto("TP Physique Quantique", "Physique", 5, "pending"),
            new AssignmentItemDto("Projet Algorithmique", "Informatique", 7, "pending"),
            new AssignmentItemDto("Exercices Algèbre", "Mathématiques", -3, "done"),
            new AssignmentItemDto("Rapport de Chimie", "Chimie", -7, "done")
    );

    public List<AssignmentItemDto> listAssignments(String status, String search) {
        return seed.stream()
                .filter(a -> status == null || status.isBlank() || "all".equalsIgnoreCase(status) || a.getStatus().equalsIgnoreCase(status))
                .filter(a -> {
                    if (search == null || search.isBlank()) return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return a.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || a.getCourse().toLowerCase(Locale.ROOT).contains(q);
                })
                .collect(Collectors.toList());
    }
}
