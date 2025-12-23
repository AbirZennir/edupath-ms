package com.example.dto.ai;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
public class DashboardAnalyticsDTO {
    private Stats stats;
    private List<AtRiskStudent> atRiskStudents;
    private List<ProfileDistribution> profileDistribution;
    private List<Evolution> evolution;

    @JsonProperty("moduleSuccess")
    private List<ModuleSuccessDTO> moduleSuccess;

    @Data
    @NoArgsConstructor
    public static class Stats {
        private int studentsAtRisk;
        private int totalStudents;
        private double successRate;
        private double engagementAvg;
        private int resourcesConsulted;
    }

    @Data
    @NoArgsConstructor
    public static class AtRiskStudent {
        private int idStudent;
        private String name;
        private String className;
        private double riskScore;
        private String lastConnection;
        private String status;
    }

    @Data
    @NoArgsConstructor
    public static class ProfileDistribution {
        private String name;
        private int value;
        private String color;
    }

    @Data
    @NoArgsConstructor
    public static class Evolution {
        private String week;
        private double success;
        private double engagement;
    }
}
