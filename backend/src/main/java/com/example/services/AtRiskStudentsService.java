package com.example.services;

import com.example.dto.AtRiskStudentDto;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Service
public class AtRiskStudentsService {

    private final RestTemplate restTemplate;

    @Autowired
    public AtRiskStudentsService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${fastapi.url}")
    private String fastapiUrl;

    public List<AtRiskStudentDto> getAtRiskStudents() {
        String url = fastapiUrl + "/api/at-risk-students";
        AtRiskStudentDto[] response = restTemplate.getForObject(url, AtRiskStudentDto[].class);
        return response != null ? List.of(response) : List.of();
    }
}
