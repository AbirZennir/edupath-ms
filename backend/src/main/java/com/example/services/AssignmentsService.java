package com.example.services;

import com.example.dto.AssignmentItemDto;
import com.example.entities.learning.Assessment;
import com.example.entities.learning.StudentAssessment;
import com.example.repositories.AssessmentRepository;
import com.example.repositories.StudentAssessmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class AssignmentsService {

    @Autowired
    private AssessmentRepository assessmentRepository;

    @Autowired
    private StudentAssessmentRepository studentAssessmentRepository;

    private final List<AssignmentItemDto> seed = List.of(
            new AssignmentItemDto("Devoir Mathématiques Ch.5", "Mathématiques", 2, "pending"),
            new AssignmentItemDto("TP Physique Quantique", "Physique", 5, "pending"),
            new AssignmentItemDto("Projet Algorithmique", "Informatique", 7, "pending"),
            new AssignmentItemDto("Exercices Algèbre", "Mathématiques", -3, "done"),
            new AssignmentItemDto("Rapport de Chimie", "Chimie", -7, "done"));

    public List<AssignmentItemDto> listAssignments(String status, String search) {
        return seed.stream()
                .filter(a -> status == null || status.isBlank() || "all".equalsIgnoreCase(status)
                        || a.getStatus().equalsIgnoreCase(status))
                .filter(a -> {
                    if (search == null || search.isBlank())
                        return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return a.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || a.getCourse().toLowerCase(Locale.ROOT).contains(q);
                })
                .collect(Collectors.toList());
    }

    public List<AssignmentItemDto> getStudentAssignments(Integer studentId, String status, String search) {
        List<Assessment> allAssessments = assessmentRepository.findAll();
        List<StudentAssessment> studentAssessments = studentAssessmentRepository.findByStudentId(studentId);

        if (allAssessments.isEmpty()) {
            return listAssignments(status, search);
        }

        Set<Integer> completedIds = studentAssessments.stream()
                .filter(sa -> sa.getScore() > 0)
                .map(sa -> sa.getStudentAssessmentId().getId_assessment())
                .collect(Collectors.toSet());

        LocalDate today = LocalDate.now();
        int currentDay = (int) ChronoUnit.DAYS.between(LocalDate.of(2024, 1, 1), today);

        return allAssessments.stream()
                .map(a -> {
                    int daysLeft = a.getDate() - currentDay;
                    boolean isDone = completedIds.contains(a.getId_assessment());
                    String assignmentStatus = isDone ? "done" : (daysLeft < 0 ? "late" : "pending");

                    return new AssignmentItemDto(
                            formatTitle(a.getAssessment_type()),
                            a.getCode_module() != null ? a.getCode_module() : "Cours",
                            daysLeft,
                            assignmentStatus);
                })
                .filter(a -> status == null || status.isBlank() || "all".equalsIgnoreCase(status)
                        || a.getStatus().equalsIgnoreCase(status))
                .filter(a -> {
                    if (search == null || search.isBlank())
                        return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return a.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || a.getCourse().toLowerCase(Locale.ROOT).contains(q);
                })
                .sorted(Comparator.comparingInt(AssignmentItemDto::getDueInDays))
                .limit(50)
                .collect(Collectors.toList());
    }

    private String formatTitle(String type) {
        if (type == null)
            return "Devoir";
        switch (type.toUpperCase()) {
            case "TMA":
                return "Devoir à rendre";
            case "CMA":
                return "Évaluation continue";
            case "EXAM":
                return "Examen";
            default:
                return type;
        }
    }
}
