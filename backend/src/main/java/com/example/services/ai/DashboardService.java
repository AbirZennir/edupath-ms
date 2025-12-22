package com.example.services.ai;

import com.example.dto.ai.DashboardAnalyticsDTO;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class DashboardService {

    private final RestTemplate restTemplate;

    @Value("${fastapi.url}")
    private String fastapiUrl;

    public DashboardService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public DashboardAnalyticsDTO getDashboardAnalytics() {
        String url = fastapiUrl + "/dashboard/analytics";
        return restTemplate.getForObject(url, DashboardAnalyticsDTO.class);
    }
}

