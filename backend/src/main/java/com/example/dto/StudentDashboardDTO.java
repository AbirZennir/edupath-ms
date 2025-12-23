package com.example.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class StudentDashboardDTO {
    private StudentInfo student;
    private ProgressInfo progression;
    private List<CourseProgress> courses;
    private List<UrgentAssignment> urgentAssignments;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class StudentInfo {
        private String nom;
        private String prenom;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ProgressInfo {
        private int global;
        private int weekChange;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CourseProgress {
        private String title;
        private int progress;
        private String icon;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class UrgentAssignment {
        private String title;
        private int daysLeft;
    }
}
