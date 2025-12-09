package com.example.services;

import com.example.dto.AssignmentDto;
import com.example.dto.CourseProgressDto;
import com.example.dto.DashboardDto;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DashboardService {

    public DashboardDto getDashboardForStudent(int studentId) {
        // Pour l’instant on renvoie des données mockées qui correspondent à l’écran mobile.
        // On pourra plus tard brancher sur les entités StudentInfo / StudentAssessment.
        String name = "Sophie";
        double generalProgress = 0.78; // 78 %
        double deltaWeek = 0.05;       // +5 % cette semaine

        List<CourseProgressDto> courses = List.of(
                new CourseProgressDto("Mathématiques", 0.75),
                new CourseProgressDto("Physique", 0.60),
                new CourseProgressDto("Informatique", 0.85)
        );

        List<AssignmentDto> urgentAssignments = List.of(
                new AssignmentDto("Devoir Mathématiques Ch.5", "À rendre bientôt", 2),
                new AssignmentDto("TP Physique Quantique", "À rendre bientôt", 5)
        );

        return new DashboardDto(name, generalProgress, deltaWeek, courses, urgentAssignments);
    }
}
