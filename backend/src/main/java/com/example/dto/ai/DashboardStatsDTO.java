package com.example.dto.ai;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DashboardStatsDTO {
    private int studentsAtRisk;
    private int totalStudents;
    private double successRate;
    private double engagementAvg;
    private long resourcesConsulted;
}
