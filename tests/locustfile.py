from locust import HttpUser, task, between

class WMSChatbotUser(HttpUser):
    wait_time = between(2, 4)

    @task
    def test_query_endpoint(self):
        headers = {"Content-Type": "application/json"}
        payload = {"user_question": "list top 3 open picklist"}
        self.client.post("/query", json=payload, headers=headers)