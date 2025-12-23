package com.example.services;

import com.example.dto.GradeItemDto;
import com.example.dto.GradesDto;
import com.example.entities.learning.StudentAssessment;
import com.example.repositories.StudentAssessmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class GradesService {

    @Autowired
    private StudentAssessmentRepository studentAssessmentRepository;

    public GradesDto getGradesForStudent(int studentId) {
        List<StudentAssessment> assessments = studentAssessmentRepository.findByStudentId(studentId);

        if (assessments.isEmpty()) {
            // Fallback to mock data
            return getMockGrades();
        }

        List<GradeItemDto> grades = assessments.stream()
                .filter(sa -> sa.getScore() > 0)
                .map(sa -> {
                    String course = sa.getAssessment() != null ? sa.getAssessment().getCode_module() : "Cours";
                    double score = sa.getScore();
                    return new GradeItemDto(course, score);
                })
                .collect(Collectors.toList());

        double average = grades.stream()
                .mapToDouble(GradeItemDto::getGrade)
                .average()
                .orElse(0.0);

        return new GradesDto(average, grades);
    }

    private GradesDto getMockGrades() {
        List<GradeItemDto> grades = List.of(
                new GradeItemDto("Mathématiques", 15.5),
                new GradeItemDto("Physique", 14.0),
                new GradeItemDto("Informatique", 17.0),
                new GradeItemDto("Chimie", 13.5));

        double average = grades.stream()
                .mapToDouble(GradeItemDto::getGrade)
                .average()
                .orElse(0.0);

        return new GradesDto(average, grades);
    }
}
