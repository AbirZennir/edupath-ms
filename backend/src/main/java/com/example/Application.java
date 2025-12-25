package com.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.HashMap;
import java.util.Map;

@SpringBootApplication
public class Application {

	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(Application.class);
		Map<String, Object> props = new HashMap<>();
		props.put("server.port", 8082);
		app.setDefaultProperties(props);
		app.run(args);
	}

}
