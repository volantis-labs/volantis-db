import CVolantisDB

public extension VolantisError {
    var codeName: String {
        get throws {
            guard let name = ze_error_code_name(rawValue) else {
                throw VolantisError.internalError
            }
            return String(cString: name)
        }
    }
}
