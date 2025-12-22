package com.example.services;

import com.example.dto.AtRiskStudentDto;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.client.RestTemplate;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

public class AtRiskStudentsServiceTest {

    @Mock
    private RestTemplate restTemplate;

    @InjectMocks
    private AtRiskStudentsService atRiskStudentsService;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        ReflectionTestUtils.setField(atRiskStudentsService, "fastapiUrl", "http://localhost:8000");
    }

    @Test
    public void testGetAtRiskStudents() {
        AtRiskStudentDto student = new AtRiskStudentDto();
        student.setId(1);
        student.setNom("Test Student");
        student.setNiveau("élevé");

        AtRiskStudentDto[] response = new AtRiskStudentDto[] { student };

        when(restTemplate.getForObject("http://localhost:8000/api/at-risk-students", AtRiskStudentDto[].class))
                .thenReturn(response);

        List<AtRiskStudentDto> result = atRiskStudentsService.getAtRiskStudents();

        assertEquals(1, result.size());
        assertEquals("Test Student", result.get(0).getNom());
    }
}
