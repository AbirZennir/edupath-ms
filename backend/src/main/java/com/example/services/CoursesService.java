package com.example.services;

import com.example.dto.CourseItemDto;
import com.example.entities.learning.Courses;
import com.example.repositories.CoursesRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

@Service
public class CoursesService {

    @Autowired
    private CoursesRepository coursesRepository;
    
    @Autowired
    private com.example.repositories.StudentInfoRepository studentInfoRepository;

    public List<CourseItemDto> listCourses(String status, String search) {
        return coursesRepository.findAll().stream()
                .map(course -> {
                    String id = course.getCourseId().getCode_module() + "_" + course.getCourseId().getCode_presentation();
                    String codeModule = course.getCourseId().getCode_module();
                    String codePresentation = course.getCourseId().getCode_presentation();
                    
                    // Mock stats for now as requested
                    int studentsCount = (int) (Math.random() * 500) + 10;
                    int successRate = (int) (Math.random() * 40) + 60;
                    int completion = (int) (Math.random() * 30) + 70;

                    return new CourseItemDto(
                            id,
                            codeModule,
                            codePresentation,
                            codeModule + " - " + codePresentation, // Title fallback
                            studentsCount,
                            successRate,
                            completion
                    );
                })
                .filter(c -> {
                    if (search == null || search.isBlank()) return true;
                    String q = search.toLowerCase(Locale.ROOT);
                    return c.getTitle().toLowerCase(Locale.ROOT).contains(q)
                            || c.getCodeModule().toLowerCase(Locale.ROOT).contains(q)
                            || c.getCodePresentation().toLowerCase(Locale.ROOT).contains(q);
                })
                .collect(Collectors.toList());
    }

    public List<Courses> getAllCourses() {
       return coursesRepository.findAll();
    }

    public List<com.example.dto.StudentDto> getStudentsForCourse(String courseId) {
        // Expected format: AAA_2013J
        String[] parts = courseId.split("_");
        if (parts.length != 2) return List.of();
        
        String codeModule = parts[0];
        String codePresentation = parts[1];

        return studentInfoRepository.findByCode_moduleAndCode_presentation(codeModule, codePresentation).stream()
                .map(s -> {
                    com.example.dto.StudentDto dto = new com.example.dto.StudentDto();
                    dto.setId(s.getId_student());
                    dto.setNom("Student " + s.getId_student()); // Mock name
                    dto.setGender(s.getGender());
                    dto.setRegion(s.getRegion());
                    dto.setFinalResult(s.getFinal_result());
                    dto.setAvatar(dto.getNom().substring(0, 1)); // First letter
                    
                    // Mock stats based on result
                    String res = s.getFinal_result().toLowerCase();
                    if (res.contains("distinction")) {
                        dto.setProfil("Très performant");
                        dto.setReussite(90 + (int)(Math.random() * 10));
                        dto.setEngagement(90 + (int)(Math.random() * 10));
                    } else if (res.contains("pass")) {
                        dto.setProfil("Assidu");
                        dto.setReussite(60 + (int)(Math.random() * 20));
                        dto.setEngagement(70 + (int)(Math.random() * 20));
                    } else if (res.contains("fail")) {
                        dto.setProfil("En difficulté");
                        dto.setReussite(20 + (int)(Math.random() * 20));
                        dto.setEngagement(30 + (int)(Math.random() * 30));
                    } else {
                         dto.setProfil("Procrastinateur");
                         dto.setReussite(40 + (int)(Math.random() * 20));
                         dto.setEngagement(40 + (int)(Math.random() * 20));
                    }
                    
                    return dto;
                })
                .collect(Collectors.toList());
    }
}
