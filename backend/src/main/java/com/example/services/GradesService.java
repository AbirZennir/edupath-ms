package com.example.services;

import com.example.dto.GradeItemDto;
import com.example.dto.GradesDto;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class GradesService {

    public GradesDto getGradesForStudent(int studentId) {
        // Données mockées alignées sur l'écran mobile.
        List<GradeItemDto> grades = List.of(
                new GradeItemDto("Mathématiques", 15.5),
                new GradeItemDto("Physique", 14.0),
                new GradeItemDto("Informatique", 17.0),
                new GradeItemDto("Chimie", 13.5)
        );

        double average = grades.stream()
                .mapToDouble(GradeItemDto::getGrade)
                .average()
                .orElse(0.0);

        return new GradesDto(average, grades);
    }
}
