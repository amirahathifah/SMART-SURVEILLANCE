import nexmo

client = nexmo.Client(key='89e9e927', secret='CZTGYEP1jgWn7gIa')

client.send_message({
    'from': 'Vonage SMS API',
    'to': '60106692394',
    'text': 'unknown person has been detected',
})