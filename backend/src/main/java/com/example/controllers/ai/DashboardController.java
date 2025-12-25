package com.example.controllers.ai;

import com.example.dto.ai.DashboardAnalyticsDTO;
import com.example.services.ai.DashboardService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/dashboard")
public class DashboardController {

    private final DashboardService dashboardService;

    public DashboardController(DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    @GetMapping("/analytics")
    public ResponseEntity<DashboardAnalyticsDTO> getDashboardAnalytics() {
        DashboardAnalyticsDTO analytics = dashboardService.getDashboardAnalytics();
        return ResponseEntity.ok(analytics);
    }
}
