package com.example.services;

import com.example.dto.AtRiskStudentDto;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestTemplate;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.client.ExpectedCount.once;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

public class AtRiskStudentsServiceTest {

    private RestTemplate restTemplate;
    private MockRestServiceServer mockServer;
    private AtRiskStudentsService service;

    @BeforeEach
    public void setUp() {
        restTemplate = new RestTemplate();
        mockServer = MockRestServiceServer.createServer(restTemplate);
        service = new AtRiskStudentsService(restTemplate);
        ReflectionTestUtils.setField(service, "fastapiUrl", "http://localhost:8000");
    }

    @Test
    public void testArrayOfObjectsMapping() {
        String json = "[ {\"id\": 1, \"nom\": \"Alice\", \"classe\": \"2021A\", \"modules\": \"Math,CS\", \"risque\": 2 } ]";
        mockServer.expect(once(), requestTo("http://localhost:8000/ai/at-risk-students"))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        List<AtRiskStudentDto> list = service.getAtRiskStudents();
        mockServer.verify();
        assertNotNull(list);
        assertEquals(1, list.size());
        AtRiskStudentDto dto = list.get(0);
        assertEquals(1, dto.getId());
        assertEquals("Alice", dto.getNom());
        assertEquals("2021A", dto.getClasse());
        assertEquals("Math,CS", dto.getModules());
        assertEquals(2, dto.getRisque());
    }

    @Test
    public void testArrayOfArraysMapping() {
        String json = "[ [1, \"Alice\", \"2021A\", \"Math,CS\", 2, null, null, null] ]";
        mockServer.expect(once(), requestTo("http://localhost:8000/ai/at-risk-students"))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        List<AtRiskStudentDto> list = service.getAtRiskStudents();
        mockServer.verify();
        assertNotNull(list);
        assertEquals(1, list.size());
        AtRiskStudentDto dto = list.get(0);
        assertEquals(1, dto.getId());
        assertEquals("Alice", dto.getNom());
        assertEquals("2021A", dto.getClasse());
        assertEquals("Math,CS", dto.getModules());
        assertEquals(2, dto.getRisque());
    }

    @Test
    public void testRealWorldMapping() {
        // Based on debug output
        String json = "[{\"idStudent\":79378,\"name\":\"Student 79378\",\"className\":\"BBB 2013B\",\"riskScore\":100,\"lastConnection\":\"N/A\",\"status\":\"critical\"}]";
        mockServer.expect(once(), requestTo("http://localhost:8000/ai/at-risk-students"))
                .andRespond(withSuccess(json, MediaType.APPLICATION_JSON));

        List<AtRiskStudentDto> list = service.getAtRiskStudents();
        mockServer.verify();
        assertNotNull(list);
        assertEquals(1, list.size());
        AtRiskStudentDto dto = list.get(0);
        assertEquals(79378, dto.getId());
        assertEquals("Student 79378", dto.getNom());
        assertEquals("BBB 2013B", dto.getClasse());
        assertEquals(100, dto.getRisque());
        assertEquals("elevé", dto.getNiveau()); // Translated from critical
        assertEquals("BBB", dto.getModules()); // Extracted from BBB 2013B
    }
}
