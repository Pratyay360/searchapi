import { defineRailway, project, service } from "railway/iac";

export default defineRailway(() => {
	const app = service("searchapi", {
		host: "0.0.0.0",
		port: "8000",
		image: "ghcr.io/pratyay360/searchapi:latest",
	});

	return project("searchapi", {
		resources: [app],
	});
});
